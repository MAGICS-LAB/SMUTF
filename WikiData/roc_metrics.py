# compute auc based on similarity matrix values

import os
import pandas as pd
from sklearn.metrics import roc_auc_score

def pair_matches(sub_dir):
    src_f = sub_dir+"/Table1.csv"
    tar_f = sub_dir+"/Table2.csv"
    df1 = pd.read_csv(src_f)
    df2 = pd.read_csv(tar_f)
    df1_cols = df1.columns
    df2_cols = df2.columns
    gnd_truths = open(sub_dir+"/mapping.txt", 'r').readlines()
    gnd_truths_pairs = set()
    for p in gnd_truths:
        plst = p.split(', ')
        # for MovieLens & HDX
        gnd_truths_pairs.add((plst[0][1:], plst[1][0:-2]))
    # print(gnd_truths_pairs)

    # get similarity matrix
    probs_f = sub_dir+"/similarity_matrix_value.csv"
    df = pd.read_csv(probs_f)
    tars = df.columns.to_list()
    tars = tars[1:]
    srcs = df.iloc[:, 0].to_list()
    num_pairs = len(tars)*len(srcs)
    ys, probs = [], []
    for tar in tars:
        tcol = df[tar].tolist()
        for i in range(len(tcol)):
            if (srcs[i], tar) in gnd_truths_pairs:
                ys.append(1)
            else:
                ys.append(0)
            probs.append(tcol[i])

    auc_score = roc_auc_score(ys, probs)

    return auc_score

def infer(tar_dir):
    aucs = []
    for x in os.walk(tar_dir):
        dir_name = x[0]
        if dir_name != tar_dir:
            print(dir_name)
            cur = pair_matches(dir_name)
            aucs.append(cur)
            print("auc: ", cur)
    res_auc = sum(aucs)/len(aucs)
    print("macro-AUC: ", res_auc)

if __name__ == "__main__":
    infer(".")