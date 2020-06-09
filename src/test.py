import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN,AgglomerativeClustering,Birch,KMeans
import numpy as np
import sys
import os
import pandas as pd
from utils.extract_features import save_relation,dump_data,load_data,generate_pair,tanimoto
from model import MetaPathGenerator
from utils.Metric import f1_score
from utils.ConfigLoader import _load_param
import json
import json5
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import trange,tqdm
from sklearn import preprocessing


class Tester(object):
    def __init__(self,args):
        self.args = args
        self.test_pub_data = json.load (open (args['test_pub_data_path'], 'r', encoding='utf-8'))
        self.test_author_data = json.load (open (args['test_row_data_path'], 'r', encoding='utf-8'))

        self.val_pub_data = json.load (open (args['val_pub_data_path'], 'r', encoding='utf-8'))
        self.val_author_data = json.load (open (args['val_row_data_path'], 'r', encoding='utf-8'))

        self.val_global_features = pd.read_pickle (args['feature_global_val_path'])
        self.test_global_features = pd.read_pickle (args['feature_global_test_path'])

        self.val_local_features = pd.read_pickle (args['feature_local_val_path'])
        self.test_local_features = pd.read_pickle (args['feature_local_test_path'])

        self.val_labels_data = json.load (open (args['val_label_path'], 'r', encoding='utf-8'))

    def do_DBCAN_eval(self):
        for w in np.arange(0.8,1.1,0.01):
            for eps in np.arange(0,1,1):
                result = {}
                for name in self.val_author_data:
                    real_counter = len(self.val_labels_data[name])

                    #print(real_counter)
                    pubs = []
                    for clusters in self.val_author_data[name]:
                        pubs.append (clusters)

                    local_features = []
                    global_features = []
                    cp = set ()
                    for i,pid in enumerate(pubs):
                        if np.sum(self.val_local_features[pid]) == 0:
                            cp.add(i)
                        local_features.append(self.val_local_features[pid])
                        global_features.append(self.val_global_features[pid])

                    global_features = np.array(global_features)
                    local_features = np.array(local_features)

                    local_features = pairwise_distances(local_features,metric="cosine")
                    global_features = pairwise_distances (global_features, metric="cosine")
                   # w = 0.7
                    sim = (local_features + w*global_features)/(1+w)

                    #pre = DBSCAN (eps=eps, min_samples=3, metric="precomputed").fit_predict (sim)
                    pre = AgglomerativeClustering(n_clusters=real_counter,affinity="precomputed",linkage="average").fit_predict (sim)
                    #pre = KMeans(n_clusters=real_counter,precompute_distances=True).fit_predict(sim)
                    pre = np.array (pre)



                    outlier = set ()
                    for i in range (len (pre)):
                        if pre[i] == -1:
                            outlier.add (i)
                    for i in cp:
                        outlier.add (i)


                    ##基于阈值的相似性匹配
                    paper_pair = generate_pair (pubs, outlier)
                    paper_pair1 = paper_pair.copy ()
                    K = len (set (pre))
                    for i in range (len (pre)):
                        if i not in outlier:
                            continue
                        j = np.argmax (paper_pair[i])
                        while j in outlier:
                            paper_pair[i][j] = -1
                            j = np.argmax (paper_pair[i])
                        if paper_pair[i][j] >= 1.5:
                            pre[i] = pre[j]
                        else:
                            pre[i] = K
                            K = K + 1

                    for ii, i in enumerate (outlier):
                        for jj, j in enumerate (outlier):
                            if jj <= ii:
                                continue
                            else:
                                if paper_pair1[i][j] >= 1.5:
                                    pre[j] = pre[i]

                    # print (pre, len (set (pre)))

                    result[name] = []
                    for i in set (pre):
                        oneauthor = []
                        for idx, j in enumerate (pre):
                            if i == j:
                                oneauthor.append (pubs[idx])
                        result[name].append (oneauthor)

                #json.dump (result, open (self.args['val_result'], 'w', encoding='utf-8'), indent=4)
                f1 = f1_score (result, self.args)
                print ("w:%2f eps:%.2f f1:%.4f "%(w,eps,f1))

    def do_DBSCAN_test(self):
        result = {}
        for name in tqdm(self.test_author_data):
            pubs = []
            for clusters in self.test_author_data[name]:
                pubs.append (clusters)

            local_features = []
            global_features = []
            cp = set ()
            for i,pid in enumerate(pubs):
                if np.sum(self.test_local_features[pid]) == 0:
                    cp.add(i)
                local_features.append(self.test_local_features[pid])
                global_features.append(self.test_global_features[pid])

            local_features = pairwise_distances(local_features,metric="cosine")
            global_features = pairwise_distances (global_features, metric="cosine")
            w = 0.3
            sim = (local_features + w*global_features)/(1+w)

            pre = DBSCAN (eps=0.15, min_samples=3, metric="precomputed").fit_predict (sim)
            pre = np.array (pre)



            outlier = set ()
            for i in range (len (pre)):
                if pre[i] == -1:
                    outlier.add (i)
            for i in cp:
                outlier.add (i)


            ##基于阈值的相似性匹配
            paper_pair = generate_pair (pubs, outlier)
            paper_pair1 = paper_pair.copy ()
            K = len (set (pre))
            for i in range (len (pre)):
                if i not in outlier:
                    continue
                j = np.argmax (paper_pair[i])
                while j in outlier:
                    paper_pair[i][j] = -1
                    j = np.argmax (paper_pair[i])
                if paper_pair[i][j] >= 1.5:
                    pre[i] = pre[j]
                else:
                    pre[i] = K
                    K = K + 1

            for ii, i in enumerate (outlier):
                for jj, j in enumerate (outlier):
                    if jj <= ii:
                        continue
                    else:
                        if paper_pair1[i][j] >= 1.5:
                            pre[j] = pre[i]

            # print (pre, len (set (pre)))

            result[name] = []
            for i in set (pre):
                oneauthor = []
                for idx, j in enumerate (pre):
                    if i == j:
                        oneauthor.append (pubs[idx])
                result[name].append (oneauthor)

        json.dump (result, open (self.args['test_result'], 'w', encoding='utf-8'), indent=4)

if __name__ == "__main__":
    argv = sys.argv
    if len (argv) == 2:
        config_file = sys.argv[1]
        if not os.path.exists (config_file):
            print ("there is no such file please check file need to be *.json5")
            exit (0)

        config = _load_param (config_file)
        test = Tester(config)
        #test.do_DBCAN_eval()
        test.do_DBSCAN_test()