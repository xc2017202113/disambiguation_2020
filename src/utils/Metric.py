import json
import numpy as np
import pandas as pd
import random
import copy


################compute the f1 pair-wise score###############
###################output is the result's format's json args is config json file##############3
def f1_score (output, args):
    '''output format:{authorname:[cluster1],[cluster2],[cluster3]...}'''
    label_file_path = args['val_label_path']
    label = json.load (open (label_file_path, 'r', encoding='utf-8'))
    PairsCorrectlyPredictedToSameAuthor = 0
    TotalPairsPredictedToSameAuthor = 0
    TotalParisToSameAuthor = 0
    author2papers = {}
    # outputcount = {}

    for authors in label:
        author2papers[authors] = []

        for authodid in label[authors]:

            author2papers[authors].extend (label[authors][authodid])
            papers_num = len(label[authors][authodid])
            TotalParisToSameAuthor += papers_num * (papers_num - 1) / 2

    for authors in output:
        labels_authors_paper = output[authors]
        # print(labels_authors_paper)

        for each_clusters in labels_authors_paper:
            TotalPairsPredictedToSameAuthor += len (each_clusters) * (len (each_clusters) - 1) / 2

    for authors in output:
        cluster_num = len (output[authors])
        authors_cluster = output[authors]
        labels_authors = label[authors]
        for authorid in labels_authors:
            for each_clusters in authors_cluster:
                authorid_correctly = 0
                for paper in each_clusters:
                    if paper in label[authors][authorid]:
                        authorid_correctly += 1

                PairsCorrectlyPredictedToSameAuthor += authorid_correctly * (authorid_correctly - 1) / 2


    # print(PairsCorrectlyPredictedToSameAuthor)
    # print(TotalPairsPredictedToSameAuthor)
    # print(TotalParisToSameAuthor)
    if PairsCorrectlyPredictedToSameAuthor < 1:
        acc = 0
    else:
        acc = PairsCorrectlyPredictedToSameAuthor / TotalPairsPredictedToSameAuthor
    recall = PairsCorrectlyPredictedToSameAuthor / TotalParisToSameAuthor
    # print(acc)
    # print(recall)
    if acc == 0 and recall == 0:
        return 0
    f1 = 2 * (acc * recall) / (acc + recall)

    return f1

def get_label (paper_author_dict, id_i, id_j):
    if paper_author_dict[id_i] == paper_author_dict[id_j]:
        return 1
    return 0

def get_dis(x,y):
    return np.sqrt(np.sum((x-y)*(x-y)))

def check_distance(feature_dict,val_label_path):
    label = json.load (open (val_label_path, 'r', encoding='utf-8'))
    author2papers = {}
    paper_author_dict = {}
    for author in label:
        author2papers[author] = []
        for authorid in label[author]:
            for paperid in label[author][authorid]:
                paper_author_dict[paperid] = authorid
                author2papers[author].append(paperid)

    sample_num = 200
    for author in label:
        print("=============")
        print(author)
        len_paper = len (author2papers[author])
        acc = []
        for t in range(sample_num):
            i = random.randint (0, len_paper - 1)
            j = random.randint (0, len_paper - 1)

            p_i = author2papers[author][i]
            p_j = author2papers[author][j]
            label = get_label (paper_author_dict, p_i, p_j)

            k = random.randint (0, len_paper - 1)

            p_k = author2papers[author][k]
            label_ = get_label (paper_author_dict, p_i, p_k)
            while label_ == label:
                k = random.randint (0, len_paper - 1)
                p_k = author2papers[author][k]
                label_ = get_label (paper_author_dict, p_i, p_k)


            if label == 1:
                pos_dis = get_dis(feature_dict[p_i],feature_dict[p_j])
                neg_dis = get_dis(feature_dict[p_i],feature_dict[p_k])
                print("pos_dis:%.4f neg_dis:%.4f"%(pos_dis,neg_dis))
            else:
                pos_dis = get_dis (feature_dict[p_i], feature_dict[p_k])
                neg_dis = get_dis (feature_dict[p_i], feature_dict[p_j])
                print ("pos_dis:%.4f neg_dis:%.4f"%(pos_dis, neg_dis))
            acc.append(neg_dis-0.5>pos_dis)
        print("========== acc:",np.mean(acc))


if __name__ == "__main__":
    feature_dict = pd.read_pickle("feature/global_train_feature5.pickle")
    val_label_path = "data/val_author_label.json"
    check_distance(feature_dict,val_label_path)
    # true = json.load (open (val_label_path, 'r', encoding='utf-8'))
    # output = {}
    # for author in true:
    #     output[author] = []
    #     for authorid in true[author]:
    #         #print(true[author])
    #         output[author].append(true[author][authorid])
    #         # print(true[author][authorid])
    #         # print(type(true[author][authorid]))
    #         # print(len(true[author][authorid]))
    #         # exit(0)
    # output = json.load (open ("result/bert+DBSCAN_trained.json", 'r', encoding='utf-8'))
    # print(f1_score(output,{"val_label_path":val_label_path}))

