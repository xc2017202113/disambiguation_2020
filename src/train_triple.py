import torch
import numpy as np
import torch.nn as nn
import os
import sys
import json
import random
from pytorch_transformers import AdamW,WarmupLinearSchedule
import torch.optim as optim
from utils.ConfigLoader import _load_param
from utils.DataLoader import Dataloader
from utils.triplet import euclidean_distance,triplet_loss,dis_acc
from model import Global_Model
from sklearn.cluster import AgglomerativeClustering
from time import time
import pandas as pd



class Trainer(object):
    def __init__(self,args):
        self.args = args
        self.train_batch_num = args['train_batch']
        self.Dataloader = Dataloader(args)
        print("preparing the train_data")
        self.train_data = self.Dataloader.load_train_batches()
        print("preparing the val_data")

        print("train data len:",len(self.train_data)*self.train_batch_num)
        self.cuda_gpu = (torch.cuda.is_available () and args['use_gpu'])

        print("build modeling:")
        self.global_model = Global_Model(args)

        if (self.cuda_gpu):
            # torch.nn.DataParallel (self.global_model, device_ids=gpus).cuda ()
            self.global_model = self.global_model.cuda()


        self.global_optimer = AdamW (self.global_model.parameters(), lr=args['global_lr'])

        num_total_steps = len(self.train_data)*args['global_epoch']
        num_warmup_steps = int(args['global_warmup_rate']*num_total_steps)

        self.global_scheduler = WarmupLinearSchedule (self.global_optimer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    def train_global(self):
        epoches = self.args['global_epoch']
        times = time()



        #self.writeglobal_features ()
        max_acc  = 0
        for epoch in range(epoches):
            train_datas = self.Dataloader.load_train_batches()

            loss_list = []
            acc_list = []
            times = time()
            for batch in range(len(train_datas)):

                text_feature = torch.tensor(train_datas[batch]["text"])
                text_pos_feature = torch.tensor(train_datas[batch]["pos_text"])
                text_neg_feature = torch.tensor(train_datas[batch]["neg_text"])

                if self.cuda_gpu:
                    text_feature = text_feature.cuda()
                    text_pos_feature = text_pos_feature.cuda ()
                    text_neg_feature = text_neg_feature.cuda ()

                text_emb = self.global_model(text_feature)
                pos_emb = self.global_model (text_pos_feature)
                neg_emb = self.global_model (text_neg_feature)


                #print(text_emb.shape)
                pos_dis,neg_dis = euclidean_distance(text_emb,pos_emb),euclidean_distance(text_emb,neg_emb)
                pos_origin_dis,neg_origin_dis = torch.cosine_similarity(text_emb,pos_emb,dim=1),torch.cosine_similarity(text_emb,neg_emb,dim=1)
                #pos_dis,neg_dis = torch.cosine_similarity(text_emb,pos_emb,dim=1),torch.cosine_similarity(text_emb,neg_emb,dim=1)
                mean_pos_dis,mean_neg_dis = torch.mean(pos_origin_dis).detach().numpy(),torch.mean(neg_origin_dis).detach().numpy()
                acc = torch.mean(dis_acc(pos_dis,neg_dis)).detach().numpy()
                acc_list.append(acc)
                loss = triplet_loss(pos_dis,neg_dis)
                loss_np = loss
                if self.cuda_gpu:
                    loss_np = loss_np.cpu()
                loss_np = loss_np.detach().numpy()
                loss_list.append(loss_np)
                self.global_optimer.zero_grad()

                loss.backward()
                self.global_optimer.step ()
                self.global_scheduler.step ()


                if batch % 200 == 0:

                    print("batch: %d loss:%.4f acc:%.4f pos_dis:%f neg_dis:%f "%(batch,loss_np,acc,mean_pos_dis,mean_neg_dis))

            mean_acc = np.mean(acc_list)
            print("epoch:%d loss:%.4f acc:%.4f time:[%.2fs]"%(epoch,np.mean(loss_list),mean_acc,time()-times))

        torch.save(self.global_model.state_dict(),self.args['global_model_save_path'])
        self.writeglobal_features()
        print("training_complete!")

    def writeglobal_features(self):
        print("writing_train_features:")
        train_feature_data = self.Dataloader.train_features
        val_feature = self.Dataloader.val_features
        test_feauture = self.Dataloader.test_features
        train_out_path = self.args['feature_global_train_path']
        global_feature_dict = {}
        for keys in train_feature_data:
            input_tensor = np.array(train_feature_data[keys].values)
            input_tensor = torch.tensor([input_tensor])
            output_tensor = self.global_model(input_tensor)
            output_np = (output_tensor[0]).detach().numpy()
            global_feature_dict[keys] = output_np
        pd.to_pickle(global_feature_dict,train_out_path)

        val_out_path = self.args['feature_global_val_path']
        val_feature_dict = {}
        for keys in val_feature:
            input_tensor = np.array (val_feature[keys].values)
            input_tensor = torch.tensor ([input_tensor])
            output_tensor = self.global_model (input_tensor)
            output_np = (output_tensor[0]).detach ().numpy ()
            val_feature_dict[keys] = output_np
        pd.to_pickle (val_feature_dict, val_out_path)

        test_out_path = self.args['feature_global_test_path']
        test_feature_dict = {}
        for keys in test_feauture:
            input_tensor = np.array (test_feauture[keys].values)
            input_tensor = torch.tensor ([input_tensor])
            output_tensor = self.global_model (input_tensor)
            output_np = (output_tensor[0]).detach ().numpy ()
            test_feature_dict[keys] = output_np
        pd.to_pickle (test_feature_dict, test_out_path)




if __name__ == "__main__":
    argv = sys.argv
    if len (argv) == 2:
        config_file = sys.argv[1]
        if not os.path.exists (config_file):
            print ("there is no such file please check file need to be *.json5")
            exit (0)

        config = _load_param (config_file)
        # data_dir = config['data_dir']
        trainer = Trainer (config)

        trainer.train_global ()