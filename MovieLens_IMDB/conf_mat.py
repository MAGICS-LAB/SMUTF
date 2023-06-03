# compute recall, precision, f1 metrics based on similarity matrix labels

import pandas as pd
import numpy as np

def conf_mat(dir):
    gnd_truths = open(dir+"mappings.txt", 'r').readlines()
    labels = dir+"similarity_matrix_label.csv"

    gnd_truths_pairs = set()
    for p in gnd_truths:
        plst = p.split(', ')
        # print(plst)
        gnd_truths_pairs.add((plst[0][2:-1], plst[1][1:-3]))
    num_truths = len(gnd_truths_pairs)

    df = pd.read_csv(labels)
    tars = df.columns.to_list()
    tars = tars[1:]
    srcs = df.iloc[:, 0].to_list()
    num_pairs = len(tars)*len(srcs)
    pred_pairs = df[df == 1.0].stack().index.tolist()
    pred_pairs = [(srcs[p[0]], p[1]) for p in pred_pairs]
    num_preds = len(pred_pairs)
    # print(gnd_truths_pairs)

    TP = 0
    for p in pred_pairs:
        if p in gnd_truths_pairs:
            TP += 1
    FP = num_preds-TP
    TN = num_pairs-num_truths-FP
    FN = num_truths-TP

    res = {"recall": 0, "precision": 0, "F1": 0}
    res["recall"] = TP/(TP+FN)
    res["precision"] = TP/(TP+FP)
    # res["F1"] = 2*res["recall"]*res["precision"]/(res["recall"]+res["precision"])

    # print(TP, FP, TN, FN)
    # print(num_truths, TP+FN)
    return res

def compute_perf():
    rs, ps = [], []
    for i in range(2):
        dir_name = "./pair_"+str(i)+"/"
        cur = conf_mat(dir_name)
        rs.append(cur['recall'])
        ps.append(cur['precision'])
        print(dir_name, cur)
    r = sum(rs)/len(rs)
    p = sum(ps)/len(ps)
    print("recall", r)
    print("precision", p)
    print("F1", 2*r*p/(r+p))

if __name__=="__main__":
    compute_perf()