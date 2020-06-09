import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
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

class TrainRondomWalk(object):
    def __init__(self,args):
        self.args = args
        self.test_pub_data = json.load(open(args['test_pub_data_path'], 'r', encoding='utf-8'))
        self.test_author_data = json.load(open(args['test_row_data_path'], 'r', encoding='utf-8'))

        self.val_pub_data = json.load (open (args['val_pub_data_path'], 'r', encoding='utf-8'))
        self.val_author_data = json.load (open (args['val_row_data_path'], 'r', encoding='utf-8'))

        self.train_pub_data = json.load (open (args['train_pub_data_path'], 'r', encoding='utf-8'))
        self.train_author_data = json.load (open (args['train_row_data_path'], 'r', encoding='utf-8'))

        self.val_features = pd.read_pickle(args['feature_global_val_path'])
        self.test_features = pd.read_pickle(args['feature_global_test_path'])

    #train on the val dataset
    def train_val(self):
        result = {}

        for n,name in tqdm(enumerate(self.val_author_data)):

            pubs = []
            #get the author's all paper
            for clusters in self.val_author_data[name]:

                pubs.append(clusters)
            #print(pubs)

            name_pubs_raw = {}
            for i,pid in enumerate(pubs):
                name_pubs_raw[pid] = self.val_pub_data[pid]
            #load the author's features
            save_relation(name_pubs_raw,name)

            mpg = MetaPathGenerator ()
            mpg.read_data ("gene")

            all_embs = []
            rw_num = 10
            cp = set()
            #start to random walk
            for k in range(rw_num):
                mpg.generate_WMRW ("gene/RW.txt", 5, 20)
                sentences = word2vec.Text8Corpus (r'gene/RW.txt')
                ##########use word2vec to train the paper's embedding###############
                model = word2vec.Word2Vec (sentences, size=128, negative=25, min_count=1, window=10)
                embs = []
                for i, pid in enumerate (pubs):
                    if pid in model:
                        embs.append (model[pid])
                    else:
                        cp.add (i)
                        embs.append (np.zeros (128))
                all_embs.append (embs)
            all_embs = np.array (all_embs)

            ##########################loading the sematic feautures#################
            ptext_emb = load_data ('gene', 'ptext_emb.pkl')
            tcp = load_data ('gene', 'tcp.pkl')

            tembs = []
            for i, pid in enumerate (pubs):
                #tembs.append (ptext_emb[pid])
                tembs.append(self.val_features[pid])

            ##############get the paper's connection's cosine matrix####################
            sk_sim = np.zeros ((len (pubs), len (pubs)))
            for k in range (rw_num):
                sk_sim = sk_sim + pairwise_distances (all_embs[k], metric="cosine")
            sk_sim = sk_sim / rw_num

            ##############get the paper's semantic's cosine matrix####################
            tembs = pairwise_distances (tembs, metric="cosine")

            w = 0.25
            sim = (np.array (sk_sim) + w * np.array (tembs)) / (1 + w)

            pre = DBSCAN (eps=0.2, min_samples=3, metric="precomputed").fit_predict (sim)
            pre = np.array (pre)

            ##离群论文集
            outlier = set ()
            for i in range (len (pre)):
                if pre[i] == -1:
                    outlier.add (i)
            for i in cp:
                outlier.add (i)
            for i in tcp:
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

            #print (pre, len (set (pre)))

            result[name] = []
            for i in set (pre):
                oneauthor = []
                for idx, j in enumerate (pre):
                    if i == j:
                        oneauthor.append (pubs[idx])
                result[name].append (oneauthor)

        json.dump (result, open (self.args['val_result'], 'w', encoding='utf-8'), indent=4)
        f1 = f1_score(result,self.args)
        print("f1:",f1)

    ###train on test dataset#####
    def train_test(self):
        result = {}

        for n,name in tqdm(enumerate(self.test_author_data)):

            pubs = []
            for clusters in self.test_author_data[name]:

                pubs.append(clusters)
            #print(pubs)

            name_pubs_raw = {}
            for i,pid in enumerate(pubs):
                name_pubs_raw[pid] = self.test_pub_data[pid]
            save_relation(name_pubs_raw,name)

            mpg = MetaPathGenerator ()
            mpg.read_data ("gene")

            all_embs = []
            rw_num = 10
            cp = set()
            for k in range(rw_num):
                mpg.generate_WMRW ("gene/RW.txt", 5, 20)
                sentences = word2vec.Text8Corpus (r'gene/RW.txt')
                model = word2vec.Word2Vec (sentences, size=128, negative=25, min_count=1, window=10)
                embs = []
                for i, pid in enumerate (pubs):
                    if pid in model:
                        embs.append (model[pid])
                    else:
                        cp.add (i)
                        embs.append (np.zeros (128))
                all_embs.append (embs)
            all_embs = np.array (all_embs)

            ptext_emb = load_data ('gene', 'ptext_emb.pkl')
            tcp = load_data ('gene', 'tcp.pkl')

            tembs = []
            for i, pid in enumerate (pubs):
                #tembs.append (ptext_emb[pid])
                tembs.append(self.test_features[pid])


            sk_sim = np.zeros ((len (pubs), len (pubs)))
            for k in range (rw_num):
                sk_sim = sk_sim + pairwise_distances (all_embs[k], metric="cosine")
            sk_sim = sk_sim / rw_num

            tembs = pairwise_distances (tembs, metric="cosine")

            w = 0.3
            sim = (np.array (sk_sim) + w * np.array (tembs)) / (1 + w)

            pre = DBSCAN (eps=0.15, min_samples=3, metric="precomputed").fit_predict (sim)
            pre = np.array (pre)

            ##离群论文集
            outlier = set ()
            for i in range (len (pre)):
                if pre[i] == -1:
                    outlier.add (i)
            for i in cp:
                outlier.add (i)
            for i in tcp:
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

            #print (pre, len (set (pre)))

            result[name] = []
            for i in set (pre):
                oneauthor = []
                for idx, j in enumerate (pre):
                    if i == j:
                        oneauthor.append (pubs[idx])
                result[name].append (oneauthor)

        json.dump (result, open (self.args['test_result'], 'w', encoding='utf-8'), indent=4)

    ###################loading the feature into files to do triple loss training###############
    def load_train_features(self):
        print("start to dump train_features")

        features = {}
        out_feature_path = self.args['feature_train_path']
        for name in tqdm(self.train_author_data):
            pubs = []
            for authorid in self.train_author_data[name]:

                for clusters in self.train_author_data[name][authorid]:
                    pubs.append (clusters)
                # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.train_pub_data[pid]
            save_relation (name_pubs_raw, name)
            ptext_emb = load_data ('gene', 'ptext_emb.pkl')

            tembs = []
            for i, pid in enumerate (pubs):
                # tembs.append (ptext_emb[pid])
                features[pid] = ptext_emb[pid]
        train_dataframe = pd.DataFrame(features)
        train_dataframe.to_pickle(out_feature_path)

    def load_valid_features(self):
        print ("start to dump valid_features")

        features = {}
        out_feature_path = self.args['feature_val_path']
        for name in tqdm (self.val_author_data):

            pubs = []
            for clusters in self.val_author_data[name]:
                pubs.append (clusters)
            # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.val_pub_data[pid]
            save_relation (name_pubs_raw, name)
            ptext_emb = load_data ('gene', 'ptext_emb.pkl')

            tembs = []
            for i, pid in enumerate (pubs):
                # tembs.append (ptext_emb[pid])
                features[pid] = ptext_emb[pid]
        valid_dataframe = pd.DataFrame (features)
        valid_dataframe.to_pickle (out_feature_path)

    def load_test_features(self):
        print ("start to dump test_features")

        features = {}
        out_feature_path = self.args['feature_test_path']
        for name in tqdm (self.test_author_data):

            pubs = []
            for clusters in self.test_author_data[name]:
                pubs.append (clusters)
            # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.test_pub_data[pid]
            save_relation (name_pubs_raw, name)
            ptext_emb = load_data ('gene', 'ptext_emb.pkl')

            tembs = []
            for i, pid in enumerate (pubs):
                # tembs.append (ptext_emb[pid])
                features[pid] = ptext_emb[pid]
        valid_dataframe = pd.DataFrame (features)
        valid_dataframe.to_pickle (out_feature_path)

    def load_train_local_features(self):
        print ("start to dump train_features")

        local_feaures = {}
        out_feature_path = self.args['feature_local_train_path']
        for name in tqdm (self.train_author_data):
            pubs = []
            for authorid in self.train_author_data[name]:

                for clusters in self.train_author_data[name][authorid]:
                    pubs.append (clusters)
                # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.train_pub_data[pid]
            save_relation (name_pubs_raw, name)
            mpg = MetaPathGenerator ()
            mpg.read_data ("gene")


            rw_num = 10
            cp = set ()

            for k in range (rw_num):
                mpg.generate_WMRW ("gene/RW.txt", 5, 20)
                sentences = word2vec.Text8Corpus (r'gene/RW.txt')
                model = word2vec.Word2Vec (sentences, size=128, negative=25, min_count=1, window=10)
                #embs = []
                for i, pid in enumerate (pubs):
                    if pid in model:
                        embs =  model[pid]
                    else:
                        cp.add (i)
                        embs = np.zeros (128)

                    if pid not in local_feaures:
                        local_feaures[pid] = [embs]
                    else:
                        local_feaures[pid].append(embs)

        for pid in local_feaures:
            local_feaures[pid] = np.array(local_feaures[pid])
            local_feaures[pid] = np.mean(local_feaures[pid],axis=0)

        train_Dataframe = pd.DataFrame(local_feaures)
        train_Dataframe.to_pickle(out_feature_path)

    def load_valid_local_features(self):
        print ("start to dump valid_features")

        local_feaures = {}
        out_feature_path = self.args['feature_local_val_path']
        for name in tqdm (self.val_author_data):

            pubs = []
            for clusters in self.val_author_data[name]:
                pubs.append (clusters)
            # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.val_pub_data[pid]
            save_relation (name_pubs_raw, name)
            mpg = MetaPathGenerator ()
            mpg.read_data ("gene")

            rw_num = 10
            cp = set ()

            for k in range (rw_num):
                mpg.generate_WMRW ("gene/RW.txt", 5, 20)
                sentences = word2vec.Text8Corpus (r'gene/RW.txt')
                model = word2vec.Word2Vec (sentences, size=128, negative=25, min_count=1, window=10)
                # embs = []
                for i, pid in enumerate (pubs):
                    if pid in model:
                        embs = model[pid]
                    else:
                        cp.add (i)
                        embs = np.zeros (128)

                    if pid not in local_feaures:
                        local_feaures[pid] = [embs]
                    else:
                        local_feaures[pid].append (embs)

        for pid in local_feaures:
            local_feaures[pid] = np.array (local_feaures[pid])
            local_feaures[pid] = np.mean (local_feaures[pid], axis=0)

        train_Dataframe = pd.DataFrame (local_feaures)
        train_Dataframe.to_pickle (out_feature_path)

    def load_test_local_features(self):
        print ("start to dump test_features")

        local_feaures = {}
        out_feature_path = self.args['feature_local_test_path']
        for name in tqdm (self.test_author_data):

            pubs = []
            for clusters in self.test_author_data[name]:
                pubs.append (clusters)
            # print(pubs)

            name_pubs_raw = {}
            for i, pid in enumerate (pubs):
                name_pubs_raw[pid] = self.test_pub_data[pid]
            save_relation (name_pubs_raw, name)
            mpg = MetaPathGenerator ()
            mpg.read_data ("gene")

            rw_num = 10
            cp = set ()

            for k in range (rw_num):
                mpg.generate_WMRW ("gene/RW.txt", 5, 20)
                sentences = word2vec.Text8Corpus (r'gene/RW.txt')
                model = word2vec.Word2Vec (sentences, size=128, negative=25, min_count=1, window=10)
                # embs = []
                for i, pid in enumerate (pubs):
                    if pid in model:
                        embs = model[pid]
                    else:
                        cp.add (i)
                        embs = np.zeros (128)

                    if pid not in local_feaures:
                        local_feaures[pid] = [embs]
                    else:
                        local_feaures[pid].append (embs)

        for pid in local_feaures:
            local_feaures[pid] = np.array (local_feaures[pid])
            local_feaures[pid] = np.mean (local_feaures[pid], axis=0)

        train_Dataframe = pd.DataFrame (local_feaures)
        train_Dataframe.to_pickle (out_feature_path)

if __name__ == "__main__":
    argv = sys.argv
    if len (argv) == 2:
        config_file = sys.argv[1]
        if not os.path.exists (config_file):
            print ("there is no such file please check file need to be *.json5")
            exit (0)

        config = _load_param (config_file)
        trainer = TrainRondomWalk(config)
        # trainer.load_train_features()
        # trainer.load_test_features()
        # trainer.load_valid_features()
        # trainer.load_test_local_features ()
        # trainer.load_train_local_features()
        # trainer.load_valid_local_features()

        trainer.train_test()
        #trainer.train_val()