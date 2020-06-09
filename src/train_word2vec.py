import json5
import json
import re
import numpy as np
from utils.ConfigLoader import _load_param
from gensim.models import word2vec
import sys
import os

class Train_Word2Vec(object):
    def __init__(self,args):
        self.train_pub_data = json.load(open(args['train_all_pub_path'], 'r', encoding='utf-8'))
        self.test_pub_data = json.load(open(args['test_pub_data_path'], 'r', encoding='utf-8'))

        self.all_text_path = args['all_text_path']

        self.args = args

        #############load train and test's all text to file in order to train word2vec###############
    def load_text(self):
        #stop word#
        r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
        f1 = open(self.all_text_path,'w',encoding='utf-8')
        for i,pid in enumerate(self.train_pub_data):
            pub = self.train_pub_data[pid]
            for author in pub["authors"]:
                if "org" in author:
                    org = author["org"]
                    pstr = org.strip ()
                    pstr = pstr.lower ()
                    pstr = re.sub (r, ' ', pstr)
                    pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
                    f1.write (pstr + '\n')

            title = pub["title"]
            pstr = title.strip ()
            pstr = pstr.lower ()
            pstr = re.sub (r, ' ', pstr)
            pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
            f1.write (pstr + '\n')

            if "abstract" in pub and type (pub["abstract"]) is str:
                abstract = pub["abstract"]
                pstr = abstract.strip ()
                pstr = pstr.lower ()
                pstr = re.sub (r, ' ', pstr)
                pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
                f1.write (pstr + '\n')

            venue = pub["venue"]
            pstr = venue.strip ()
            pstr = pstr.lower ()
            pstr = re.sub (r, ' ', pstr)
            pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
            f1.write (pstr + '\n')

        for i,pid in enumerate(self.test_pub_data):
            pub = self.test_pub_data[pid]
            for author in pub["authors"]:
                if "org" in author:
                    org = author["org"]
                    pstr = org.strip ()
                    pstr = pstr.lower ()
                    pstr = re.sub (r, ' ', pstr)
                    pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
                    f1.write (pstr + '\n')

            title = pub["title"]
            pstr = title.strip ()
            pstr = pstr.lower ()
            pstr = re.sub (r, ' ', pstr)
            pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
            f1.write (pstr + '\n')

            if "abstract" in pub and type (pub["abstract"]) is str:
                abstract = pub["abstract"]
                pstr = abstract.strip ()
                pstr = pstr.lower ()
                pstr = re.sub (r, ' ', pstr)
                pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
                f1.write (pstr + '\n')

            venue = pub["venue"]
            pstr = venue.strip ()
            pstr = pstr.lower ()
            pstr = re.sub (r, ' ', pstr)
            pstr = re.sub (r'\s{2,}', ' ', pstr).strip ()
            f1.write (pstr + '\n')

        f1.close()

    def train_word2vec(self):
        sentences = word2vec.Text8Corpus (self.args['all_text_path'])
        model = word2vec.Word2Vec (sentences, size=128, negative=5, min_count=2, window=5)
        model.save (self.args['save_word2vec_model'])

if __name__ == "__main__":
    argv = sys.argv
    if len (argv) == 2:
        config_file = sys.argv[1]
        if not os.path.exists (config_file):
            print ("there is no such file please check file need to be *.json5")
            exit (0)

        config = _load_param (config_file)
        model = Train_Word2Vec(config)
        print("loading the text")
        model.load_text()
        print("training_model-")
        model.train_word2vec()