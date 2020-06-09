import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
import math
import random
from tqdm import trange,tqdm
import pandas as pd

from  pytorch_transformers import BertTokenizer

class Dataloader(object):
    def __init__(self,args):
        self.train_row_data_path = args['train_row_data_path']


        self.train_features = pd.read_pickle(args['feature_train_path'])
        self.test_features = pd.read_pickle(args['feature_test_path'])
        self.val_features = pd.read_pickle (args['feature_val_path'])
        
        self.args = args

        self.train_data = json.load(open(self.train_row_data_path, 'r', encoding='utf-8'))

        self.train_pub_data = json.load (open (args['train_pub_data_path'], 'r', encoding='utf-8'))
        self.train_data_len = len(self.train_data)


        print("get_pairs")
        self.all_paper = self.get_all_papers()
        self.pairs = self.gen_train_data(self.train_data,self.train_pub_data)

        #self.val_label_path = args['val_label_path']


    def get_all_papers(self):
        all_paper = []
        for paper in self.train_pub_data:
            all_paper.append(paper)
        return all_paper



    def get_label (self,paper_author_dict, id_i, id_j):
        if paper_author_dict[id_i] == paper_author_dict[id_j]:
            return 1
        return 0

    #generate pair-wise sample
    def gen_train_data (self,train_data,train_pub_data):
        # pos_dict = {}
        # neg_dict = {}

        paper_author_dict = {}
        author_papers = {}
        for author in train_data:
            author_papers[author] = []
            for author_id in train_data[author]:
                for p_id in train_data[author][author_id]:
                    #if not offer the paper's info we can't do train so we drop these papers
                    if p_id in train_pub_data.keys():
                        author_papers[author].append(p_id)
                        #papers.append(p_id)
                        paper_author_dict[p_id] = author_id + author


        pairs = []
        sample_num = self.args['each_author_sample_pair_num']

        for author in tqdm(train_data):
            len_paper = len(author_papers[author])
            #print(author)


            for k in range(sample_num):
                i = random.randint(0,len_paper-1)
                j = random.randint(0,len_paper-1)

                p_i = author_papers[author][i]
                p_j = author_papers[author][j]
                label = self.get_label (paper_author_dict, p_i, p_j)
                while label == 0:
                    j = random.randint (0, len_paper - 1)
                    p_j = author_papers[author][j]
                    label = self.get_label (paper_author_dict, p_i, p_j)


                m = random.randint(0,len(self.all_paper)-1)
                p_k = self.all_paper[m]
                while self.get_label (paper_author_dict, p_i, p_k) == 1:
                    m = random.randint (0, len (self.all_paper) - 1)
                    p_k = self.all_paper[m]
                pairs.append([(p_i,p_j,1),(p_i,p_k,0)])





        return pairs

    def load_train_batches(self):
        #self.pairs = self.gen_train_data (self.train_data, self.train_pub_data)
        train_batches = self.args['train_batch']

        train_data_batches = []
        for i in range(int(np.ceil(len(self.pairs)/train_batches))):
            min_index = i*train_batches
            max_index = min(len(self.pairs),(i+1)*train_batches)
            #train_data_batches.append(pairs[min_index:max_index])
            texts = []
            text_pos = []
            text_neg = []
            #label = []
            for j in range(max_index-min_index):
                #pairs[j+min_index]:[(text,pos_text,1),(text,neg_text,0)]

                paperid = self.pairs[j+min_index][0][0]
                pos_paperid = self.pairs[j+min_index][0][1]
                neg_paperid = self.pairs[j+min_index][1][1]



                text_features =  np.array(self.train_features[paperid].values)
                text_pos_features = np.array(self.train_features[pos_paperid].values)
                text_neg_features = np.array (self.train_features[neg_paperid].values)


                texts.append(text_features)
                text_pos.append(text_pos_features)
                text_neg.append(text_neg_features)


            train_data_batches.append({"text":np.array(texts),
                                       "pos_text":np.array(text_pos),
                                       "neg_text":np.array(text_neg)})

        return train_data_batches


