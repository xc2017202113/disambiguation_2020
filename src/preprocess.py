import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
import math
import random
from tqdm import trange,tqdm

###############this file aims to process the origin file#########################
###############1.depart the train data to train set and valid set################
###############2.if some paper not in pub file,then simple delete it#############


train_row_data_path = 'data/OAG-v2-track1/OAG-v2-track1/train_author.json'
train_pub_data_path = 'data/OAG-v2-track1/OAG-v2-track1/train_pub.json'

test_row_data_path = 'data/OAG-v2-track1/OAG-v2-track1/valid/sna_valid_author_raw.json'
test_pub_data_path = 'data/OAG-v2-track1/OAG-v2-track1/valid/sna_valid_pub.json'

out_train_row_data_path = 'data/train_author.json'
out_train_pub_data_path = 'data/train_pub.json'
out_val_row_data_path = 'data/val_author.json'
out_val_row_data_label_path = 'data/val_author_label.json'
out_val_pub_data_path = 'data/val_pub.json'
out_test_data = 'data/test_author.json'
out_test_pub_data = 'data/test_pub.json'

depart_rate = 0.8

train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
train_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))
train_authors = [author for author in train_data]
#train_pub_authors = [author for author in train_pub_data]
print(len(train_authors))
#print(len(train_pub_authors))



train_len = int(len(train_authors)*depart_rate)
val_len  = len(train_authors) - train_len
train_data_dict = {}
train_pub_data_dict = {}

print("start to dump the train data:")
random.shuffle(train_authors)

for i in trange(train_len):
    author_name = train_authors[i]
    train_data_dict[author_name] = {}
    for authorid in train_data[author_name]:
        train_data_dict[author_name][authorid] = []
        for paperid in train_data[author_name][authorid]:
            if paperid in train_pub_data.keys ():

                train_pub_data_dict[paperid] = train_pub_data[paperid]
                train_data_dict[author_name][authorid].append(paperid)
            else:
                print ("missing train id:", paperid)

json.dump(train_data_dict, open (out_train_row_data_path, 'w', encoding='utf-8'), indent=4)
json.dump(train_pub_data_dict, open (out_train_pub_data_path, 'w', encoding='utf-8'), indent=4)

train_data_dict = {}
train_pub_data_dict = {}
val_data_dict = {}
print("start to dump the val data:")
for i in trange(val_len):
    author_name = train_authors[i+train_len]
    train_data_dict[author_name] = {}
    val_data_dict[author_name] = []
    for authorid in train_data[author_name]:
        train_data_dict[author_name][authorid] = []
        for paperid in train_data[author_name][authorid]:
            if paperid in train_pub_data.keys():
                train_data_dict[author_name][authorid].append(paperid)
                val_data_dict[author_name].append (paperid)
                train_pub_data_dict[paperid] = train_pub_data[paperid]
            else:
                print ("missing val id:", paperid)

json.dump(train_data_dict, open (out_val_row_data_label_path, 'w', encoding='utf-8'), indent=4)
json.dump(val_data_dict, open (out_val_row_data_path, 'w', encoding='utf-8'), indent=4)
json.dump(train_pub_data_dict, open (out_val_pub_data_path, 'w', encoding='utf-8'), indent=4)

test_data_dict = {}
test_pub_dict = json.load(open(test_pub_data_path, 'r', encoding='utf-8'))
test_row_data = json.load(open(test_row_data_path, 'r', encoding='utf-8'))
print("start to dump the test data:")
for authors in tqdm(test_row_data):
    test_data_dict[authors] = []
    for paperid in test_row_data[authors]:
        if paperid in test_pub_dict:
            test_data_dict[authors].append(paperid)
        else:
            print("test paperid missing:",paperid)

json.dump(test_pub_dict, open (out_test_pub_data, 'w', encoding='utf-8'), indent=4)
json.dump(test_data_dict, open (out_test_data, 'w', encoding='utf-8'), indent=4)