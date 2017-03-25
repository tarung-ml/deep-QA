from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util import ConfusionMatrix, Progbar, minibatches

from evaluate import exact_match_score, f1_score
from evaluate import evaluate


logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class QASystem(object):
    def __init__(self, FLAGS, pretrained_embeddings, vocab_dim, *args):
        self.train_dir = FLAGS.train_dir
        self.pretrained_embeddings = pretrained_embeddings

        self.FLAGS = FLAGS
        self.batch_size = FLAGS.batch_size
        self.QMAXLEN = FLAGS.QMAXLEN
        self.PMAXLEN = FLAGS.PMAXLEN
        self.embedding_size = FLAGS.embedding_size # length of word-vectors
        self.lstm_units = FLAGS.lstm_units
        self.vocab_dim = vocab_dim
        self.pretrained_embeddings = pretrained_embeddings
        self.model_output = self.FLAGS.train_dir + "/model.weights"

        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            if self.FLAGS.model_type == 'MPCM' or self.FLAGS.model_type == 'MPCM_p100' or self.FLAGS.model_type == 'MPCM_2':
                self.setup_system_MPCM()
            if self.FLAGS.model_type == 'COATT':
                self.setup_system_COATT()
            if self.FLAGS.model_type == 'COATT_mix':
                self.setup_system_COATT_mix()
            if self.FLAGS.model_type == 'COATT_fixed' or self.FLAGS.model_type == 'COATT_fixed_200':
                self.setup_system_COATT_fixed()
            if self.FLAGS.model_type == 'MPCM_fixed':
                self.setup_system_MPCM_fixed()
            if self.FLAGS.model_type == 'COATT_fixed_mix' or self.FLAGS.model_type == 'COATT_fixed_200_mix':
                self.setup_system_COATT_fixed_mix()
            self.setup_loss_enriched()
            optimizer = get_optimizer(self.FLAGS.optimizer)(self.FLAGS.learning_rate) #.minimize(self.loss)
            variables = tf.trainable_variables()
            print([v.name for v in variables])
            gradients = optimizer.compute_gradients(self.loss, variables)
            gradients = [tup[0] for tup in gradients]
            if FLAGS.clip_gradients:
                gradients, norms = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
            self.grad_norm = tf.global_norm(gradients)
            grads_and_vars = zip(gradients, variables)
            self.train_op = optimizer.apply_gradients(grads_and_vars)

    ############## TARUN'S SETUP_SYSTEM #####################
    def test_softmax(self):
        a = tf.constant(np.array([[[.1, .3, .5, .9], [0, 0, 0, 0]], [[.1, .3, .5, .9], [0, 0, 0, 0]]]))
        with tf.Session() as s:
            s.run(tf.nn.softmax(a, dim=1))
        # this should be column-wise per example in batch

            # ref: https://piazza.com/class/iw9g8b9yxp46s8?cid=2106 for example attention mechanism

    def attention_layer(self, pp, qq):
        # pp is B-by-PMAXLEN-by-2h_dim, qq is B-by-QMAXLEN-by-2h_dim
        # below will return B-by-QMAXLEN-by-PMAXLEN
        # i.e. use dot-product scoring
        s = tf.matmul(qq, tf.transpose(pp, perm=[0, 2,
                                                 1]))  # much more complexity needed here (for example softmax scaling etc.)
        s_max = tf.reduce_max(s, axis = 1, keep_dims  = True)
        s_min= tf.reduce_min(s, axis = 1, keep_dims  = True)
        s_mean = tf.reduce_mean(s, axis = 1, keep_dims  = True)
        s_enrich = tf.concat([s_max, s_min, s_mean], 1)

        # print(s.get_shape())
        alphap = tf.nn.softmax(s, dim = 1) # should be column-wise as sum(alpha_i) per paragraph-word is 1
        # Q*P
        alphaq = tf.nn.softmax(tf.transpose(s, perm = [0,2,1]), dim = 1) # should be column-wise as sum(alpha_i) per question-word is 1
        # P*Q

        #print(alpha.get_shape()); print(qq.get_shape())
        # Now produce the context vector c for paragpraph words
        cp = tf.matmul(tf.transpose(qq, perm = [0, 2, 1]), alphap) # paragraph-context vector
        # Now produce the context vector c for question words
        cq = tf.matmul(tf.transpose(pp, perm = [0, 2, 1]), alphaq) # quesiton-context vector


        # Add filter layer
        filterLayer = False
        if filterLayer:
            normed_p = tf.nn.l2_normalize(pp, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(qq, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            p_emb_p = pp

        q_concat = tf.concat([cq, tf.transpose(qq, perm = [0, 2, 1])], 1)

        c_d = tf.matmul(q_concat, alphap) # 2h*p
        p_concat = tf.concat([tf.transpose(p_emb_p, perm = [0, 2, 1]), c_d], 1)


        return p_concat #tf.concat([s, s_enrich, tf.transpose(p_emb_p, perm = [0, 2, 1]) ], 1) #c, tf.transpose(pp, perm = [0, 2, 1]),

    def setup_system_COATT(self):
        # define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        A_start = self.PMAXLEN  # maxlength of one-hot vector for  start-index
        A_end = self.PMAXLEN  # maxlength of one-hot vector for  end-index
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units

        # define placeholders
        self.q = tf.placeholder(tf.int32, [None, self.QMAXLEN], name="question")
        self.p = tf.placeholder(tf.int32, [None, self.PMAXLEN], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_end")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # add embeddings for question and paragraph
        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        # https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        #self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d
        #self.p_emb = tf.nn.dropout(self.p_emb, self.dropout)

        # print stuff to understand appropriate dims
        # pick out an LSTM cell
        cell_p_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_fwd, output_keep_prob=self.dropout)
        cell_q_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_fwd, output_keep_prob=self.dropout)
        cell_p_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_bwd, output_keep_prob=self.dropout)
        cell_q_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_bwd, output_keep_prob=self.dropout)

        # get bilstm encodings
        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])

        print(("type1", (self.p_emb).get_shape()))
        # build the hidden representation for the question (fwd and bwd and stack them together)
        with tf.variable_scope("encode_q"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_q, (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_q_fwd,
                                                                                                cell_bw=cell_q_bwd,
                                                                                                inputs=self.q_emb,
                                                                                                sequence_length=q_seq_len,
                                                                                                dtype=tf.float32)
            self.qq = tf.concat(outputs_q, 2)  # 2h_dim dimensional representation over each word in question
            # self.qq = tf.reshape(output_state_fw_q, shape = [self.batch_size, 1, 2*h_dim]) + tf.reshape(output_state_bw_q, shape = [self.batch_size, 1, 2*h_dim])  # try only using "end representation"  as question summary vector
            print(("type11", (self.qq).get_shape()))
        with tf.variable_scope("encode_p"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_p_fwd, cell_bw=cell_p_bwd,
                                                           initial_state_fw=output_state_fw_q,
                                                           inputs=self.p_emb, sequence_length=p_seq_len,
                                                           dtype=tf.float32)
            self.pp = tf.concat(outputs_p,
                                2)  # 2h_dim dimensional representation over each word in context-paragraph

        # need to mix qq and pp to get an attention matrix (question-words)-by-(paragraph-words) dimensional heat-map like matrix for each example
        # this attention matrix will allow us to localize the interesting parts of the paragraph (they will peak) and help us identify the patch corresponding to answer
        # visually the patch will ideally start at start-index and decay near end-index
        self.att = self.attention_layer(self.pp, self.qq)

        #  predictions obtain by applying softmax over something (attention vals - should be something like dim(question-words)-by-dim(paragraph)
        # currently a B-by-QMAXLEN-by-PMAXLEN tensor
        dim_att = int(self.att.get_shape()[
                          1])  # self.QMAXLEN # # first dim of something, second dim of soemthing should be self.AMAXLEN i.e. self.PMAXLEN i.e. attention computed for each word in paragraph

        print(("type2", (self.att).get_shape()))

        # apply another LSTM layer before softmax (choice of uni vs bi-directional)s
        biLayer = True
        seq_len_final = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64)) #tf.placeholder(tf.int32, [None])
        cell_final = tf.contrib.rnn.BasicLSTMCell(dim_att, state_is_tuple=True)
        if biLayer:
            out_lstm, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_final, cell_bw=cell_final,
                                                           inputs=tf.transpose(self.att, perm = [0, 2, 1]), sequence_length=seq_len_final, dtype=tf.float32)
            lstm_final = tf.concat(out_lstm, 2)
            #lstm_final = tf.transpose(, perm = [0, 2, 1])  # 2*2h_dim dimensional representation over each word in context-paragraph
        else:
            out_lstm, _ = (tf.nn.dynamic_rnn(cell=cell_final, inputs=tf.transpose(self.att, perm = [0, 2, 1]), dtype=tf.float32))
            #lstm_final = tf.transpose(out_lstm, perm=[0, 2, 1])
        print(lstm_final .get_shape())
        #lstm_final = tf.transpose(self.att, perm=[0, 2, 1])
        lstm_final = tf.nn.dropout(lstm_final, self.dropout)


        dim_final_layer = int(lstm_final.get_shape()[2])
        final_layer_ = tf.reshape(lstm_final, shape=[-1, dim_final_layer])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_start = (
        tf.reshape(tf.matmul(final_layer_, W_start), shape=[cur_batch_size, self.PMAXLEN]) + b_start)
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_end = (tf.reshape(tf.matmul(final_layer_, W_end), shape=[cur_batch_size, self.PMAXLEN]) + b_end)
        self.yp_end = tf.nn.softmax(self.logits_end)

    def setup_system_COATT_fixed_mix(self):
        # define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        A_start = self.PMAXLEN  # maxlength of one-hot vector for  start-index
        A_end = self.PMAXLEN  # maxlength of one-hot vector for  end-index
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units

        # define placeholders
        self.q = tf.placeholder(tf.int32, [None, self.QMAXLEN], name="question")
        self.p = tf.placeholder(tf.int32, [None, self.PMAXLEN], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_end")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # add embeddings for question and paragraph
        self.embedding_mat = tf.constant(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        # https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        #self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d
        #self.p_emb = tf.nn.dropout(self.p_emb, self.dropout)

        # Add filter layer
        filterLayer = True
        if filterLayer:
            normed_p = tf.nn.l2_normalize(self.p_emb, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(self.q_emb, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            p_emb_p = self.p_emb

        self.p_emb = p_emb_p

        # print stuff to understand appropriate dims
        # pick out an LSTM cell
        cell_p_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_fwd, output_keep_prob=self.dropout)
        cell_q_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_fwd, output_keep_prob=self.dropout)
        cell_p_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_bwd, output_keep_prob=self.dropout)
        cell_q_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_bwd, output_keep_prob=self.dropout)

        # get bilstm encodings
        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])

        print(("type1", (self.p_emb).get_shape()))
        # build the hidden representation for the question (fwd and bwd and stack them together)
        with tf.variable_scope("encode_q"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_q, (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_q_fwd,
                                                                                                cell_bw=cell_q_bwd,
                                                                                                inputs=self.q_emb,
                                                                                                sequence_length=q_seq_len,
                                                                                                dtype=tf.float32)
            self.qq = tf.concat(outputs_q, 2)  # 2h_dim dimensional representation over each word in question
            # self.qq = tf.reshape(output_state_fw_q, shape = [self.batch_size, 1, 2*h_dim]) + tf.reshape(output_state_bw_q, shape = [self.batch_size, 1, 2*h_dim])  # try only using "end representation"  as question summary vector
            print(("type11", (self.qq).get_shape()))
        with tf.variable_scope("encode_p"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_p_fwd, cell_bw=cell_p_bwd,
                                                           initial_state_fw=output_state_fw_q,
                                                           inputs=self.p_emb, sequence_length=p_seq_len,
                                                           dtype=tf.float32)
            self.pp = tf.concat(outputs_p,
                                2)  # 2h_dim dimensional representation over each word in context-paragraph

        # need to mix qq and pp to get an attention matrix (question-words)-by-(paragraph-words) dimensional heat-map like matrix for each example
        # this attention matrix will allow us to localize the interesting parts of the paragraph (they will peak) and help us identify the patch corresponding to answer
        # visually the patch will ideally start at start-index and decay near end-index
        self.att = self.attention_layer(self.pp, self.qq)

        #  predictions obtain by applying softmax over something (attention vals - should be something like dim(question-words)-by-dim(paragraph)
        # currently a B-by-QMAXLEN-by-PMAXLEN tensor
        dim_att = int(self.att.get_shape()[
                          1])  # self.QMAXLEN # # first dim of something, second dim of soemthing should be self.AMAXLEN i.e. self.PMAXLEN i.e. attention computed for each word in paragraph

        print(("type2", (self.att).get_shape()))

        # apply another LSTM layer before softmax (choice of uni vs bi-directional)s
        biLayer = True
        seq_len_final = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64)) #tf.placeholder(tf.int32, [None])
        cell_final = tf.contrib.rnn.BasicLSTMCell(dim_att, state_is_tuple=True)
        if biLayer:
            out_lstm, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_final, cell_bw=cell_final,
                                                           inputs=tf.transpose(self.att, perm = [0, 2, 1]), sequence_length=seq_len_final, dtype=tf.float32)
            lstm_final = tf.concat(out_lstm, 2)
            #lstm_final = tf.transpose(, perm = [0, 2, 1])  # 2*2h_dim dimensional representation over each word in context-paragraph
        else:
            out_lstm, _ = (tf.nn.dynamic_rnn(cell=cell_final, inputs=tf.transpose(self.att, perm = [0, 2, 1]), dtype=tf.float32))
            #lstm_final = tf.transpose(out_lstm, perm=[0, 2, 1])
        print(lstm_final .get_shape())
        #lstm_final = tf.transpose(self.att, perm=[0, 2, 1])
        lstm_final = tf.nn.dropout(lstm_final, self.dropout)


        dim_final_layer = int(lstm_final.get_shape()[2])
        final_layer_ = tf.reshape(lstm_final, shape=[-1, dim_final_layer])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_start = (
        tf.reshape(tf.matmul(final_layer_, W_start), shape=[cur_batch_size, self.PMAXLEN]) + b_start)
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_end = (tf.reshape(tf.matmul(final_layer_, W_end), shape=[cur_batch_size, self.PMAXLEN]) + b_end)
        self.yp_end = tf.nn.softmax(self.logits_end)

    def setup_system_COATT_mix(self):
        # define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        A_start = self.PMAXLEN  # maxlength of one-hot vector for  start-index
        A_end = self.PMAXLEN  # maxlength of one-hot vector for  end-index
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units

        # define placeholders
        self.q = tf.placeholder(tf.int32, [None, self.QMAXLEN], name="question")
        self.p = tf.placeholder(tf.int32, [None, self.PMAXLEN], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_end")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # add embeddings for question and paragraph
        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        # https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        #self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d
        #self.p_emb = tf.nn.dropout(self.p_emb, self.dropout)

        # Add filter layer
        filterLayer = True
        if filterLayer:
            normed_p = tf.nn.l2_normalize(self.p_emb, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(self.q_emb, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            p_emb_p = self.p_emb

        self.p_emb = p_emb_p

        # print stuff to understand appropriate dims
        # pick out an LSTM cell
        cell_p_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_fwd, output_keep_prob=self.dropout)
        cell_q_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_fwd, output_keep_prob=self.dropout)
        cell_p_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_bwd, output_keep_prob=self.dropout)
        cell_q_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_bwd, output_keep_prob=self.dropout)

        # get bilstm encodings
        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])

        print(("type1", (self.p_emb).get_shape()))
        # build the hidden representation for the question (fwd and bwd and stack them together)
        with tf.variable_scope("encode_q"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_q, (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_q_fwd,
                                                                                                cell_bw=cell_q_bwd,
                                                                                                inputs=self.q_emb,
                                                                                                sequence_length=q_seq_len,
                                                                                                dtype=tf.float32)
            self.qq = tf.concat(outputs_q, 2)  # 2h_dim dimensional representation over each word in question
            # self.qq = tf.reshape(output_state_fw_q, shape = [self.batch_size, 1, 2*h_dim]) + tf.reshape(output_state_bw_q, shape = [self.batch_size, 1, 2*h_dim])  # try only using "end representation"  as question summary vector
            print(("type11", (self.qq).get_shape()))
        with tf.variable_scope("encode_p"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_p_fwd, cell_bw=cell_p_bwd,
                                                           initial_state_fw=output_state_fw_q,
                                                           inputs=self.p_emb, sequence_length=p_seq_len,
                                                           dtype=tf.float32)
            self.pp = tf.concat(outputs_p,
                                2)  # 2h_dim dimensional representation over each word in context-paragraph

        # need to mix qq and pp to get an attention matrix (question-words)-by-(paragraph-words) dimensional heat-map like matrix for each example
        # this attention matrix will allow us to localize the interesting parts of the paragraph (they will peak) and help us identify the patch corresponding to answer
        # visually the patch will ideally start at start-index and decay near end-index
        self.att = self.attention_layer(self.pp, self.qq)

        #  predictions obtain by applying softmax over something (attention vals - should be something like dim(question-words)-by-dim(paragraph)
        # currently a B-by-QMAXLEN-by-PMAXLEN tensor
        dim_att = int(self.att.get_shape()[
                          1])  # self.QMAXLEN # # first dim of something, second dim of soemthing should be self.AMAXLEN i.e. self.PMAXLEN i.e. attention computed for each word in paragraph

        print(("type2", (self.att).get_shape()))

        # apply another LSTM layer before softmax (choice of uni vs bi-directional)s
        biLayer = True
        seq_len_final = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64)) #tf.placeholder(tf.int32, [None])
        cell_final = tf.contrib.rnn.BasicLSTMCell(dim_att, state_is_tuple=True)
        if biLayer:
            out_lstm, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_final, cell_bw=cell_final,
                                                           inputs=tf.transpose(self.att, perm = [0, 2, 1]), sequence_length=seq_len_final, dtype=tf.float32)
            lstm_final = tf.concat(out_lstm, 2)
            #lstm_final = tf.transpose(, perm = [0, 2, 1])  # 2*2h_dim dimensional representation over each word in context-paragraph
        else:
            out_lstm, _ = (tf.nn.dynamic_rnn(cell=cell_final, inputs=tf.transpose(self.att, perm = [0, 2, 1]), dtype=tf.float32))
            #lstm_final = tf.transpose(out_lstm, perm=[0, 2, 1])
        print(lstm_final .get_shape())
        #lstm_final = tf.transpose(self.att, perm=[0, 2, 1])
        lstm_final = tf.nn.dropout(lstm_final, self.dropout)


        dim_final_layer = int(lstm_final.get_shape()[2])
        final_layer_ = tf.reshape(lstm_final, shape=[-1, dim_final_layer])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_start = (
        tf.reshape(tf.matmul(final_layer_, W_start), shape=[cur_batch_size, self.PMAXLEN]) + b_start)
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_end = (tf.reshape(tf.matmul(final_layer_, W_end), shape=[cur_batch_size, self.PMAXLEN]) + b_end)
        self.yp_end = tf.nn.softmax(self.logits_end)

    def setup_system_COATT_fixed(self):
        # define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        A_start = self.PMAXLEN  # maxlength of one-hot vector for  start-index
        A_end = self.PMAXLEN  # maxlength of one-hot vector for  end-index
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units

        # define placeholders
        self.q = tf.placeholder(tf.int32, [None, self.QMAXLEN], name="question")
        self.p = tf.placeholder(tf.int32, [None, self.PMAXLEN], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, self.PMAXLEN], name="answer_end")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # add embeddings for question and paragraph
        self.embedding_mat = tf.constant(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        # https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        #self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d
        #self.p_emb = tf.nn.dropout(self.p_emb, self.dropout)

        # print stuff to understand appropriate dims
        # pick out an LSTM cell
        cell_p_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_fwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_q_bwd = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell_p_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_fwd, output_keep_prob=self.dropout)
        cell_q_fwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_fwd, output_keep_prob=self.dropout)
        cell_p_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_p_bwd, output_keep_prob=self.dropout)
        cell_q_bwd = tf.contrib.rnn.DropoutWrapper(cell=cell_q_bwd, output_keep_prob=self.dropout)

        # get bilstm encodings
        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])

        print(("type1", (self.p_emb).get_shape()))
        # build the hidden representation for the question (fwd and bwd and stack them together)
        with tf.variable_scope("encode_q"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_q, (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_q_fwd,
                                                                                                cell_bw=cell_q_bwd,
                                                                                                inputs=self.q_emb,
                                                                                                sequence_length=q_seq_len,
                                                                                                dtype=tf.float32)
            self.qq = tf.concat(outputs_q, 2)  # 2h_dim dimensional representation over each word in question
            # self.qq = tf.reshape(output_state_fw_q, shape = [self.batch_size, 1, 2*h_dim]) + tf.reshape(output_state_bw_q, shape = [self.batch_size, 1, 2*h_dim])  # try only using "end representation"  as question summary vector
            print(("type11", (self.qq).get_shape()))
        with tf.variable_scope("encode_p"):
            # https://www.tensorflow.org/versions/master/api_docs/python/nn/recurrent_neural_networks#bidirectional_dynamic_rnn
            outputs_p, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_p_fwd, cell_bw=cell_p_bwd,
                                                           initial_state_fw=output_state_fw_q,
                                                           inputs=self.p_emb, sequence_length=p_seq_len,
                                                           dtype=tf.float32)
            self.pp = tf.concat(outputs_p,
                                2)  # 2h_dim dimensional representation over each word in context-paragraph

        # need to mix qq and pp to get an attention matrix (question-words)-by-(paragraph-words) dimensional heat-map like matrix for each example
        # this attention matrix will allow us to localize the interesting parts of the paragraph (they will peak) and help us identify the patch corresponding to answer
        # visually the patch will ideally start at start-index and decay near end-index
        self.att = self.attention_layer(self.pp, self.qq)

        #  predictions obtain by applying softmax over something (attention vals - should be something like dim(question-words)-by-dim(paragraph)
        # currently a B-by-QMAXLEN-by-PMAXLEN tensor
        dim_att = int(self.att.get_shape()[
                          1])  # self.QMAXLEN # # first dim of something, second dim of soemthing should be self.AMAXLEN i.e. self.PMAXLEN i.e. attention computed for each word in paragraph

        print(("type2", (self.att).get_shape()))

        # apply another LSTM layer before softmax (choice of uni vs bi-directional)s
        biLayer = True
        seq_len_final = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64)) #tf.placeholder(tf.int32, [None])
        cell_final = tf.contrib.rnn.BasicLSTMCell(dim_att, state_is_tuple=True)
        if biLayer:
            out_lstm, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_final, cell_bw=cell_final,
                                                           inputs=tf.transpose(self.att, perm = [0, 2, 1]), sequence_length=seq_len_final, dtype=tf.float32)
            lstm_final = tf.concat(out_lstm, 2)
            #lstm_final = tf.transpose(, perm = [0, 2, 1])  # 2*2h_dim dimensional representation over each word in context-paragraph
        else:
            out_lstm, _ = (tf.nn.dynamic_rnn(cell=cell_final, inputs=tf.transpose(self.att, perm = [0, 2, 1]), dtype=tf.float32))
            #lstm_final = tf.transpose(out_lstm, perm=[0, 2, 1])
        print(lstm_final .get_shape())
        #lstm_final = tf.transpose(self.att, perm=[0, 2, 1])
        lstm_final = tf.nn.dropout(lstm_final, self.dropout)


        dim_final_layer = int(lstm_final.get_shape()[2])
        final_layer_ = tf.reshape(lstm_final, shape=[-1, dim_final_layer])

        # softmax layer
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_start = (
        tf.reshape(tf.matmul(final_layer_, W_start), shape=[cur_batch_size, self.PMAXLEN]) + b_start)
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([self.PMAXLEN]))
        self.logits_end = (tf.reshape(tf.matmul(final_layer_, W_end), shape=[cur_batch_size, self.PMAXLEN]) + b_end)
        self.yp_end = tf.nn.softmax(self.logits_end)


    ############## Try to enrich ANDREIB'S SETUP_SYSTEM #####################
    def setup_system_MPCM(self):
        # Define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units
        l_dim = self.FLAGS.perspective_units
        B = self.batch_size

        # Define placeholders
        self.q = tf.placeholder(tf.int32, [None, Q], name="question")
        self.p = tf.placeholder(tf.int32, [None, P], name="context_paragraph")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.p = tf.placeholder(tf.int32, [None, P], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, P], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, P], name="answer_end")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # Add embeddings for question and paragraph
        self.embedding_mat = tf.Variable(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d

        # Add filter layer
        filterLayer = True
        if filterLayer:
            normed_p = tf.nn.l2_normalize(self.p_emb, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(self.q_emb, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            self.p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            self.p_emb_p = self.p_emb

        # add dropout after filter layer
        self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        self.p_emb_p = tf.nn.dropout(self.p_emb_p, self.dropout)

        cell = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.dropout)

        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])
        # Context Representation Layer
        with tf.variable_scope("encode_q"):
            (output_q_fw, output_q_bw), (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.q_emb,
                sequence_length=q_seq_len,
                dtype=tf.float32)
        with tf.variable_scope("encode_p"):
            (output_p_fw, output_p_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.p_emb_p,
                initial_state_fw=output_state_fw_q,
                sequence_length=p_seq_len,
                dtype=tf.float32)

        # Multi-perspective context matching layer
        W1 = tf.get_variable(
            "W1",
            shape=[1, 1, h_dim, l_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(
            "W2",
            shape=[1, 1, h_dim, l_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        output_p_fw = tf.expand_dims(output_p_fw, 3)
        tp1 = tf.nn.l2_normalize(tf.multiply(output_p_fw, W1), dim=2)
        qs1 = output_q_fw[:, Q - 1, :]
        qs1 = tf.expand_dims(qs1, 1)
        qs1 = tf.expand_dims(qs1, 3)
        tq1 = tf.nn.l2_normalize(tf.multiply(qs1, W1), dim=2)
        m1 = tf.multiply(tp1, tq1)
        m1_full = tf.reduce_sum(m1, axis=2)
        m1_max = tf.reduce_max(m1, axis=2)
        m1_mean = tf.reduce_mean(m1, axis=2)


        output_p_bw = tf.expand_dims(output_p_bw, 3)
        tp2 = tf.nn.l2_normalize(tf.multiply(output_p_bw, W2), dim=2)
        qs2 = output_q_bw[:, 0, :]
        qs2 = tf.expand_dims(qs2, 1)
        qs2 = tf.expand_dims(qs2, 3)
        tq2 = tf.nn.l2_normalize(tf.multiply(qs2, W2), dim=2)
        m2 = tf.multiply(tp2, tq2)
        m2_full = tf.reduce_sum(m2, axis=2)
        m2_max = tf.reduce_max(m2, axis=2)
        m2_mean = tf.reduce_mean(m2, axis=2)

        m = tf.concat([m1_full, m1_max, m1_mean, m2_full, m2_max, m2_mean], axis=2)
        m = tf.nn.dropout(m, self.dropout)

        # Aggregation layer
        cur_batch_size = tf.shape(self.p)[0];
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P, dtype=tf.int64))
        with tf.variable_scope("mix"):
            outputs_mix, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=m,
                sequence_length=p_seq_len,
                dtype=tf.float32)
            final_layer = tf.concat(outputs_mix, 2)

        dim_final_layer = int(final_layer.get_shape()[2])
        final_layer_ = tf.reshape(final_layer, shape=[-1, dim_final_layer])

        # Prediction layer
        # this should be replaced by a NN as in the paper later
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([P]))
        mixed_start = tf.matmul(final_layer_, W_start)
        mixed_start = tf.reshape(mixed_start, shape=[-1, P])
        self.logits_start = mixed_start + b_start
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([P]))
        mixed_end = tf.matmul(final_layer_, W_end)
        mixed_end = tf.reshape(mixed_end, shape=[-1, P])
        self.logits_end = mixed_end + b_end
        self.yp_end = tf.nn.softmax(self.logits_end)


    ############## Try to enrich ANDREIB'S SETUP_SYSTEM #####################
    def setup_system_MPCM_fixed(self):
        # Define some dimensions (everything is padded)
        Q = self.QMAXLEN  # maxlength of questions
        P = self.PMAXLEN  # maxlength of context paragraphs
        V, d = self.vocab_dim, self.embedding_size  # |V|, d i.e. V words and each word-vec is d-dimensional
        h_dim = self.lstm_units
        l_dim = self.FLAGS.perspective_units
        B = self.batch_size

        # Define placeholders
        self.q = tf.placeholder(tf.int32, [None, Q], name="question")
        self.p = tf.placeholder(tf.int32, [None, P], name="context_paragraph")
        self.q_len = tf.placeholder(tf.int32, [None], name="q_len")
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.a_len = tf.placeholder(tf.float32, [None], name="answer_length")
        self.p = tf.placeholder(tf.int32, [None, P], name="context_paragraph")
        self.a_start = tf.placeholder(tf.float32, [None, P], name="answer_start")
        self.a_end = tf.placeholder(tf.float32, [None, P], name="answer_end")
        self.dropout = tf.placeholder(tf.float32, shape=())

        # Add embeddings for question and paragraph
        self.embedding_mat = tf.constant(self.pretrained_embeddings, name="pre", dtype=tf.float32)
        self.q_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.q),
                             dtype=tf.float32)  # perhaps B-by-Q-by-d
        self.p_emb = tf.cast(tf.nn.embedding_lookup(self.embedding_mat, self.p),
                             dtype=tf.float32)  # perhaps B-by-P-by-d

        # Add filter layer
        filterLayer = True
        if filterLayer:
            normed_p = tf.nn.l2_normalize(self.p_emb, dim=2) # normalize the 400 paragraph vectors
            normed_q = tf.nn.l2_normalize(self.q_emb, dim=2)
            cossim = tf.matmul(normed_p, tf.transpose(normed_q, [0, 2, 1]))
            rel = tf.reduce_max(cossim, axis = 2, keep_dims = True)
            self.p_emb_p = tf.multiply(rel, self.p_emb)
        else:
            self.p_emb_p = self.p_emb

        # add dropout after filter layer
        #self.q_emb = tf.nn.dropout(self.q_emb, self.dropout)
        #self.p_emb_p = tf.nn.dropout(self.p_emb_p, self.dropout)

        cell = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.dropout)

        cur_batch_size = tf.shape(self.p)[0];
        q_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(Q,
                                        dtype=tf.int64))  # np.ones(B) * self.QMAXLEN #tf.placeholder(tf.int32, [None])
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P,
                                        dtype=tf.int64))  # np.ones(B) * self.PMAXLEN #tf.placeholder(tf.int32, [None])
        # Context Representation Layer
        with tf.variable_scope("encode_q"):
            (output_q_fw, output_q_bw), (output_state_fw_q, output_state_bw_q) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.q_emb,
                sequence_length=q_seq_len,
                dtype=tf.float32)
        with tf.variable_scope("encode_p"):
            (output_p_fw, output_p_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.p_emb_p,
                initial_state_fw=output_state_fw_q,
                sequence_length=p_seq_len,
                dtype=tf.float32)

        # Multi-perspective context matching layer
        W1 = tf.get_variable(
            "W1",
            shape=[1, 1, h_dim, l_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(
            "W2",
            shape=[1, 1, h_dim, l_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        output_p_fw = tf.expand_dims(output_p_fw, 3)
        tp1 = tf.nn.l2_normalize(tf.multiply(output_p_fw, W1), dim=2)
        qs1 = output_q_fw[:, Q - 1, :]
        qs1 = tf.expand_dims(qs1, 1)
        qs1 = tf.expand_dims(qs1, 3)
        tq1 = tf.nn.l2_normalize(tf.multiply(qs1, W1), dim=2)
        m1 = tf.multiply(tp1, tq1)
        m1_full = tf.reduce_sum(m1, axis=2)
        m1_max = tf.reduce_max(m1, axis=2)
        m1_mean = tf.reduce_mean(m1, axis=2)


        output_p_bw = tf.expand_dims(output_p_bw, 3)
        tp2 = tf.nn.l2_normalize(tf.multiply(output_p_bw, W2), dim=2)
        qs2 = output_q_bw[:, 0, :]
        qs2 = tf.expand_dims(qs2, 1)
        qs2 = tf.expand_dims(qs2, 3)
        tq2 = tf.nn.l2_normalize(tf.multiply(qs2, W2), dim=2)
        m2 = tf.multiply(tp2, tq2)
        m2_full = tf.reduce_sum(m2, axis=2)
        m2_max = tf.reduce_max(m2, axis=2)
        m2_mean = tf.reduce_mean(m2, axis=2)

        m = tf.concat([m1_full, m1_max, m1_mean, m2_full, m2_max, m2_mean], axis=2)
        m = tf.nn.dropout(m, self.dropout)

        # Aggregation layer
        cur_batch_size = tf.shape(self.p)[0];
        p_seq_len = tf.fill(tf.expand_dims(cur_batch_size, 0),
                            tf.constant(P, dtype=tf.int64))
        with tf.variable_scope("mix"):
            outputs_mix, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=m,
                sequence_length=p_seq_len,
                dtype=tf.float32)
            final_layer = tf.concat(outputs_mix, 2)

        dim_final_layer = int(final_layer.get_shape()[2])
        final_layer_ = tf.reshape(final_layer, shape=[-1, dim_final_layer])

        # Prediction layer
        # this should be replaced by a NN as in the paper later
        W_start = tf.get_variable("W_start", shape=[dim_final_layer, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.Variable(tf.zeros([P]))
        mixed_start = tf.matmul(final_layer_, W_start)
        mixed_start = tf.reshape(mixed_start, shape=[-1, P])
        self.logits_start = mixed_start + b_start
        self.yp_start = tf.nn.softmax(self.logits_start)

        W_end = tf.get_variable("W_end", shape=[dim_final_layer, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.Variable(tf.zeros([P]))
        mixed_end = tf.matmul(final_layer_, W_end)
        mixed_end = tf.reshape(mixed_end, shape=[-1, P])
        self.logits_end = mixed_end + b_end
        self.yp_end = tf.nn.softmax(self.logits_end)






    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        # may need to do some reshaping here
        # Someone replaced yp_start with logits_start in loss. Don't really follow the change. Setting it back to original.
        with vs.variable_scope("loss"):
            self.loss_start = (tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_start, labels = tf.cast(self.a_start, tf.float32))))
            self.loss_end = (tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_end, labels = tf.cast(self.a_end, tf.float32))))
            self.loss = tf.add(self.loss_start, self.loss_end)
            pass



    # add l2 loss term for span length i.e. penalize (true_length - predicted_length)^2
    # this will heavily penalize cases where predicted_length is negative
    def setup_loss_enriched(self):
        """
        Set up your loss computation here
        :return:
        """

        # may need to do some reshaping here
        # Someone replaced yp_start with logits_start in loss. Don't really follow the change. Setting it back to original.
        with vs.variable_scope("loss"):
            self.loss_start = (tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_start, labels = tf.cast(self.a_start, tf.float32))))
            self.loss_end = (tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_end, labels = tf.cast(self.a_end, tf.float32))))
            # compute span l2 loss
            a_s_p = tf.argmax(self.yp_start, axis=1)
            a_e_p = tf.argmax(self.yp_end, axis=1)
            self.loss_span = tf.reduce_mean(tf.nn.l2_loss( tf.cast(self.a_len, tf.float32)  - tf.cast(a_s_p - a_e_p, tf.float32)))
            self.loss = tf.add(self.loss_start, self.loss_end) + self.FLAGS.span_l2*self.loss_span
            pass


    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):
        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, Q_dev, P_dev, A_start_dev, A_end_dev, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
        for sample in samples: # sample of size 100
            answers_dic = generate_answers(sess, model, sample_dataset, rev_vocab) # returns dictionary to be fed in evaluate
            result = evaluate(sample_dataset_dic, answers_dic) # takes dictionaries of form nswers[uuid] = "real answer"
            f1 += result['f1']
            em += result['exact_match']

        return f1, em

    def create_feed_dict(self, P, Q, P_len, Q_len, A_start=None, A_end=None, A_len = None, dropout=1.0):
        feed_dict = {
            self.p: P,
            self.q: Q,
            self.p_len: P_len,
            self.q_len: Q_len,
            self.dropout: dropout,
        }
        if A_start is not None:
            feed_dict[self.a_start] = A_start
        if A_end is not None:
            feed_dict[self.a_end] = A_end
        if A_len is not None:
            feed_dict[self.a_len] = A_len
        return feed_dict

    def predict_on_batch(self, sess, P, Q, P_len, Q_len):
        feed = self.create_feed_dict(P, Q, P_len, Q_len)
        (yp_start, yp_end) = sess.run([self.yp_start, self.yp_end], feed_dict=feed)
        return (yp_start, yp_end)

    def train_on_batch(self, sess, P, Q, P_len, Q_len, A_start, A_end, A_len):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(P, Q, P_len, Q_len, A_start=A_start, A_end=A_end, A_len = A_len,
                                     dropout=(1.0 - self.FLAGS.dropout))
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_examples, dev_set):
        prog = Progbar(target=1 + int(len(train_examples) / self.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.batch_size)):
            # TODO we need to remove this. Make sure your model works with variable batch sizes
            batch = batch[:7]
            if len(batch[0]) != self.batch_size:
                continue
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
        print("")

        logging.info("Evaluating on development data")
        prog = Progbar(target=1 + int(len(dev_set) / self.batch_size))
        f1 = exact_match = total = 0
        for i, batch in enumerate(minibatches(dev_set, self.batch_size)):
            # TODO we need to remove this. Make sure your model works with variable batch sizes
            if len(batch[0]) != self.batch_size:
                continue
            # Only use P and Q
            batch_pred = batch[:4]
            (ys, ye) = self.predict_on_batch(sess, *batch_pred)
            a_s = np.argmax(ys, axis=1)
            a_e = np.argmax(ye, axis=1)
            for i in range(len(a_s)):
                p_raw = batch[7][i]
                a_raw = batch[8][i]
                s = a_s[i]
                e = a_e[i]
                pred_raw = ' '.join(p_raw.split()[s:e+1])
                f1 += f1_score(pred_raw, a_raw)
                exact_match += exact_match_score(pred_raw, a_raw)
                total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        logging.info("Entity level F1/EM: %.2f/%.2f", f1, exact_match)

        return f1

    def train(self, session, train_data, dev_data):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in self.train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        start = time.time()

        saver = tf.train.Saver()

        best_score = 0.

        '''
        Proposed data format to be fed in, for each sample:
        context:
            [3,10,35,1,0,0,0,0]
            padded with 0s at the end, integers represent indexes into vocabulary
            max_context_size is the length
        context_mask:
            [True,True,True,True,False,False,False,False]
        question:
            [1,3,0,0]
            padded with 0s at the end, integers represend indexes into vocabulary
            max_question_size is the length of the list
        question_mask:
            [True,True,False,False]
        label_start:
            [1,0,0,0,0,0,0,0] - hot vector of length max_context_size
        label_end
            [0,0,0,0,1,0,0,0] - hot vector of length max_context_size

        train_examples will be a list of tuples of form:
        (context, context_mask, question, question_mask, label_start, label_end)

        embeddings matrix will be a matrix of vocabulary_size x number of word features
        '''

        for epoch in range(self.FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.FLAGS.epochs)
            score = self.run_epoch(session, train_data, dev_data)
            if score > best_score:
                best_score = score
                if saver:
                    logging.info("New best score! Saving model in %s", self.model_output)
                    saver.save(session, self.model_output)
            print("")
        logging.info("Best f1 score detected this run : %s ", best_score)
        return best_score




