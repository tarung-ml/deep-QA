import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import json
import os

from code.reader import load_data
from code.evaluate import exact_match_score, f1_score
from code.reader import load_data, load_data_home


# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')


# load predictions
predictions = json.load(open("dev-prediction.json"))
predictions = {str(k):str(v) for k,v in predictions.items()}

# load actuals
Q_test, P_test, A_start_test, A_end_test, A_len_test, P_raw_test, A_raw_test, Q_len_test, P_len_test = load_data_home("dev")
question_uuid_data = []
with open("dev.quid") as f:
    for line in f:
        question_uuid_data.append((line))

question_raw_data = []
with open("dev.question") as f:
    for line in f:
        question_raw_data.append((line))


test_data = zip(P_test, Q_test, P_len_test, Q_len_test, A_start_test, A_end_test, A_len_test, P_raw_test, A_raw_test, question_uuid_data, question_raw_data)

actuals = {} # get qkey: (q_raw, a_raw, p_raw)
for i in range(len(test_data)):
    actuals[str(test_data[i][9].strip("\n"))] = (test_data[i][10].strip("\n"), test_data[i][8].strip("\n"), test_data[i][7].strip("\n"))



# build a dataframe to store data
data = pd.DataFrame(columns = (['question', 'answer', 'paragraph', 'prediction', 'f1','em']))
i = 0

print(len(predictions.keys()))
print(len(actuals.keys()))

for key in predictions.keys():
    data.loc[i,] = [actuals[key][0], actuals[key][1], actuals[key][2], predictions[key],
                    float(f1_score(actuals[key][1], predictions[key])), exact_match_score(actuals[key][1], predictions[key])]
    i += 1
    if i % 1000 == 0:
        print(i)


# get attributes. (track trends in performance across these)
data['qtype'] = data['question'].apply(lambda x: x.split()[0].lower())

bin = 50
data['a_len'] = data['answer'].apply(lambda x: len(x.split()))
data['q_len'] = data['question'].apply(lambda x: len(x.split()))
data['p_len'] = data['paragraph'].apply(lambda x: bin*int(float(len(x.split()))/float(bin)))



# breakdown by qtype
qdata = data.groupby(['qtype'])['f1'].apply(lambda x: x.mean()*100)
qdata_em = data.groupby(['qtype'])['em'].apply(lambda x: x.mean()*100)
filename = "plots/qtype.pdf"
qtypes = ['what', 'who', 'how', 'when', 'which', 'where', 'why']
plt.xticks(range(len(qtypes)), qtypes)
plt.bar(-0.2 + np.array(range(len(qtypes))), qdata[qtypes], label = 'F1', width = 0.4, color = 'r')
plt.bar(0.2 + np.array(range(len(qtypes))), qdata_em[qtypes], label = 'EM', width = 0.4)
plt.xlabel('qtype')
plt.ylabel('Score')
plt.legend(fontsize = "small")
plt.savefig(filename); plt.clf(); plt.close();


# breakdown by alen
qdata = data.groupby(['a_len'])['f1'].apply(lambda x: x.mean()*100)
qdata_em = data.groupby(['a_len'])['em'].apply(lambda x: x.mean()*100)
filename = "plots/alens.pdf"
qtypes = range(1, 20)
plt.xticks(range(0, len(qtypes)), qtypes)
plt.bar(-0.2 + np.array(range(len(qtypes))), qdata[qtypes], label = 'F1', width = 0.4, color = 'r')
plt.bar(0.2 + np.array(range(len(qtypes))), qdata_em[qtypes], label = 'EM', width = 0.4)
plt.xlabel('answer_length')
plt.ylabel('Score')
plt.legend(fontsize = "small")
plt.savefig(filename); plt.clf(); plt.close();

# breakdown by qlen
qdata = data.groupby(['q_len'])['f1'].apply(lambda x: x.mean()*100)
qdata_em = data.groupby(['q_len'])['em'].apply(lambda x: x.mean()*100)
filename = "plots/qlens.pdf"
qtypes = range(20)
plt.xticks(range(0, len(qtypes)), qtypes)
plt.bar(-0.2 + np.array(range(len(qtypes))), qdata[qtypes], label = 'F1', width = 0.4, color = 'r')
plt.bar(0.2 + np.array(range(len(qtypes))), qdata_em[qtypes], label = 'EM', width = 0.4)
plt.xlabel('question_length')
plt.ylabel('Score')
plt.legend(fontsize = "small")
plt.savefig(filename); plt.clf(); plt.close();


# breakdown by plen
qdata = data.groupby(['p_len'])['f1'].apply(lambda x: x.mean()*100)
qdata_em = data.groupby(['p_len'])['em'].apply(lambda x: x.mean()*100)
filename = "plots/plens.pdf"
lens = data['p_len'].unique()
qtypes = sorted(lens[lens <400])
plt.xticks(range(0, len(qtypes)), [x + bin/2 for x in qtypes])
plt.bar(-0.2 + np.array(range(len(qtypes))), qdata[qtypes], label = 'F1', width = 0.4, color = 'r')
plt.bar(0.2 + np.array(range(len(qtypes))), qdata_em[qtypes], label = 'EM', width = 0.4)
plt.xlabel('context_length')
plt.ylabel('Score')
plt.legend(fontsize = "small")
plt.savefig(filename); plt.clf(); plt.close();




