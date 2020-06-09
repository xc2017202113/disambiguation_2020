import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
class MetaPathGenerator(object):
    def __init__(self):
        self.paper_author = dict ()
        self.author_paper = dict ()
        self.paper_org = dict ()
        self.org_paper = dict ()
        self.paper_conf = dict ()
        self.conf_paper = dict ()

    ##########read author's feature and write the paper's relation to files#######################
    def read_data (self, dirpath):
        temp = set ()
        with open (dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add (line)
        for line in temp:
            toks = line.strip ().split ("\t")
            if len (toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append (a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append (p)
        temp.clear ()

        with open (dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add (line)
        for line in temp:
            toks = line.strip ().split ("\t")
            if len (toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append (a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append (p)
        temp.clear ()

        with open (dirpath + "/paper_conf.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add (line)
        for line in temp:
            toks = line.strip ().split ("\t")
            if len (toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append (a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append (p)
        temp.clear ()

        # print ("#papers ", len (self.paper_conf))
        # print ("#authors", len (self.author_paper))
        # print ("#org_words", len (self.org_paper))
        # print ("#confs  ", len (self.conf_paper))

    ######generate the random walk's meta path##############
    def generate_WMRW (self, outfilename, numwalks, walklength):
        outfile = open (outfilename, 'w')
        for paper0 in self.paper_conf:
            for j in range (0, numwalks):  # wnum walks
                paper = paper0
                outline = ""
                i = 0
                while (i < walklength):
                    i = i + 1
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len (authors)
                        authorid = random.randrange (numa)
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len (papers)
                        if nump > 1:
                            paperid = random.randrange (nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange (nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len (words)
                        wordid = random.randrange (numw)
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len (papers)
                        if nump > 1:
                            paperid = random.randrange (nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange (nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                outfile.write (outline + "\n")
        outfile.close ()

class Global_Model(nn.Module):

    def __init__(self,args):
        super (Global_Model, self).__init__ ()
        self.layer1 = nn.Linear(args['emb_dim'],args['emb_dim'])
        self.layer2 = nn.Linear(args['emb_dim'],args['hidden_size'])
        #elf.layernorm = nn.LayerNorm([args['hidden_size']])

    def forward(self,input_feature,istraining=True):
        #when istraining=true, we use the bert model's dropout else dropout rate set to 0
        layer1 = F.relu(self.layer1(input_feature))
        layer2 = F.relu(self.layer2(layer1))
        out_feature = F.normalize(layer2,dim=-1)

        return out_feature

class cluster_Counter(nn.Module):
    def __init__(self,args):
        super (cluster_Counter, self).__init__ ()
        # self.BiLSTM = nn.LSTM(args['hidden_size'],int(args['hidden_size']/2),num_layers=2,batch_first=True,
        #                       dropout=args['dropout_rate'],bidirectional=True)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=1)
        self.pooling1 = nn.MaxPool2d(kernel_size=3,stride=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.pooling2 = nn.MaxPool2d (kernel_size=2, stride=2)
        self.dense = nn.Linear(38416,1)

    def forward(self,input):

        input_matrix = torch.matmul(input,input.permute(0,2,1))
        #print(input_matrix.shape)

        input_matrix.unsqueeze_(1)
        #print(input_matrix.shape)
        layer1 = F.relu(self.conv1(input_matrix))
        layer1 = self.pooling1(layer1)
        layer2 = F.relu(self.conv2(layer1))
        layer2 = self.pooling2(layer2)
        layer_size = layer2.size()
        layer_flattern = torch.reshape(layer2,(layer_size[0],layer_size[1]*layer_size[2]*layer_size[3]))
        output = F.relu(self.dense(layer_flattern)) + torch.tensor(1)
        return output