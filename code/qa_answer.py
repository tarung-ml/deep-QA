# ========= Consume path to  dev json and generate predicted json =========
# ref: http://web.stanford.edu/class/cs224n/assignment4/codalab_submission_instructions.pdf
# ref: (section 3.3) http://web.stanford.edu/class/cs224n/assignment4/assignment4.pdf
# python code/qa_answer. py -- dev_path dev.json dev-predicted.json

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

#from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

from preprocessing.squad_preprocess import read_write_dataset
from qa_data import data_to_token_ids
# for load, pad data
from reader import load_data
from keras.preprocessing.sequence import pad_sequences
# get the pad ID
from qa_data import PAD_ID
from qa_model import QASystem
from util import ConfusionMatrix, Progbar, minibatches

from evaluate import exact_match_score, f1_score

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")

## COPY THIS for final model from train.py
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")

tf.app.flags.DEFINE_integer("QMAXLEN", 60, "Max Question Length")
tf.app.flags.DEFINE_integer("PMAXLEN", 400, "Max Context Paragraph Length") # max is 766 but 99.98% have <400
tf.app.flags.DEFINE_integer("lstm_units", 100, "Number of lstm representation h_i")
tf.app.flags.DEFINE_integer("perspective_units", 50, "Number of lstm representation h_i")
tf.app.flags.DEFINE_bool("clip_gradients", True, "Do gradient clipping")
tf.app.flags.DEFINE_bool("tiny_sample", True, "Work with tiny sample")
tf.app.flags.DEFINE_float("tiny_sample_pct", 0.1, "Sample pct.")
tf.app.flags.DEFINE_float("dev_tiny_sample_pct", 1, "Sample pct.")
tf.app.flags.DEFINE_float("span_l2", 0.0001, "Span l2 loss regularization constant")


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    print("Downloading {}".format(dev_filename))
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(question_uuid_data, a_s, a_e, context_data, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    for i in range(len(a_s)):
        answer = context_data[i][a_s[i]:(a_e[i]+1)]
        answers[question_uuid_data[i]] = " ".join([rev_vocab[int(idx)] for idx in answer.split()])
    return answers


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)


    # ========= Download Dataset json =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    _, _, _ = prepare_dev(dev_dirname, dev_filename, vocab)

    # ========= Process input json =========
    prefix = os.path.join("data", "squad")

    # writes dev.answer, dev.context, dev.question, dev.span
    dev_path = FLAGS.dev_path
    dev_filename = FLAGS.dev_path.split("/")[-1]
    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', prefix)
    print("Processed {} questions and {} answers in dev".format(dev_num_questions, dev_num_answers))

    # writes dev.ids.context, dev.ids.question
    vocab_path = pjoin(os.path.join("data", "squad"), "vocab.dat")
    dev_deposit_path = pjoin(os.path.join("data", "squad"), "dev")
    x_dis_path = dev_deposit_path + ".ids.context"
    y_ids_path = dev_deposit_path + ".ids.question"
    data_to_token_ids(dev_deposit_path + ".context", x_dis_path, vocab_path)
    data_to_token_ids(dev_deposit_path + ".question", y_ids_path, vocab_path)

    # load data sets
    Q_test, P_test, A_start_test, A_end_test, A_len_test, P_raw_test, A_raw_test, Q_len_test, P_len_test = load_data(os.path.join("data", "squad"), "dev") # for our purposes this is as test set.
    question_uuid_data = []
    with open(os.path.join("data", "squad") + "/dev.quid") as f:
        for line in f:
            question_uuid_data.append((line))

    # pad the data at load-time. So, we don't need to do any masking later!!!
    # ref: https://keras.io/preprocessing/sequence/
    # if len < maxlen, pad with specified val
    # elif len > maxlen, truncate
    QMAXLEN = FLAGS.QMAXLEN
    PMAXLEN = FLAGS.PMAXLEN
    Q_test = pad_sequences(Q_test, maxlen=QMAXLEN, value=PAD_ID, padding='post')
    P_test = pad_sequences(P_test, maxlen=PMAXLEN, value=PAD_ID, padding='post')
    A_start_test = pad_sequences(A_start_test, maxlen=PMAXLEN, value=0, padding='post')
    A_end_test = pad_sequences(A_end_test, maxlen=PMAXLEN, value=0, padding='post')
    test_data = zip(P_test, Q_test, P_len_test, Q_len_test, A_start_test, A_end_test, A_len_test, P_raw_test, A_raw_test, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    with tf.Graph().as_default():
        with tf.Session() as sess:
            embeddings = np.load(FLAGS.data_dir + '/glove.trimmed.' + str(FLAGS.embedding_size) + '.npz')
            pretrained_embeddings = embeddings['glove']

            qa = QASystem(FLAGS, pretrained_embeddings, vocab_dim=len(vocab.keys()))

            initialize_model(sess, qa, train_dir)

            # get predicted start-end indices
            a_s = [] # store all start index preds
            a_e = [] # store all end index preds
            a_s_l = []
            a_e_l = []

            f1 = exact_match = total = 0; answers = {}
            prog = Progbar(target=1 + int(len(test_data) / FLAGS.batch_size))
            for i, batch in enumerate(minibatches(test_data, FLAGS.batch_size, shuffle = False)):
                batch_test =  batch[:4]
                (ys, ye) = qa.predict_on_batch(sess, *batch_test)
                a_s = (np.argmax(ys, axis=1))
                a_e = (np.argmax(ye, axis=1))
                a_s_l = a_s_l + list(a_s)
                a_e_l = a_e_l + list(a_e)

                for j in range(len(a_s)):
                    p_raw = batch[7][j]
                    a_raw = batch[8][j]
                    s = a_s[j]
                    e = a_e[j]
                    pred_raw = ' '.join(p_raw.split()[s:e + 1])
                    f1 += f1_score(pred_raw, a_raw)
                    exact_match += exact_match_score(pred_raw, a_raw)
                    total += 1
                    answers[batch[9][j].strip("\n")] = pred_raw.strip("\n")
                prog.update(i + 1, [("processed", i + 1)])
            exact_match = 100.0 * exact_match / total
            f1 = 100.0 * f1 / total
            print(("First Answer Entity level F1/EM: %.2f/%.2f", f1, exact_match))

        #answers = generate_answers(question_uuid_data, a_s_l, a_e_l, context_data, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()

