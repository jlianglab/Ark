from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch import optim as optim
import sys
import time
import math
import copy
import argparse
import os
import random
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv


# python main_linearprobing.py --gpu 2 --arch ark6_swinLarge768 --emb_path /data/ChestDR/ark6_SwinLarge_768_ChestDR/  --runs 1 --few_shot 0.05 --exp_name 5pct

def get_args_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=512,  help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--optimizer', dest='optimizer', default="adam", type=str, help="optimizer")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")
    parser.add_argument('--arch', dest='arch', default="cxr_embedding", type=str)
    parser.add_argument('--dataset', dest='dataset', default="ChestDR", type=str)
    parser.add_argument('--num_class', type=int, default=19,  help='number of classes')
    parser.add_argument('--multiclass', type=bool, default=False, help='whether multiclass task')
    parser.add_argument('--emb_path', dest='emb_path', default="/data/ChestDR/CXR_Foundation_ChestDR/", type=str)
    parser.add_argument('--emb_dim', type=int, default=1376,  help='length of embedding')
    parser.add_argument('--runs', type=int, dest='runs', default=1, help='num of runs')
    parser.add_argument('--few_shot', type=float, default=-1, help='percentage or number of samples of each class for training')
    parser.add_argument('--exp_name', dest='exp_name', default="exp_fewshot", type=str)
    parser.add_argument('--patience', type=int, dest='patience', default=10, help='num of patient epoches')
    parser.add_argument('--mode', dest='mode', default="train", type=str)
    args = parser.parse_args()
    return args


class EmbeddingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features = np.load(self.data[item])
        label = np.array(self.labels[item])
        
        return features, label

   
class ChestDR_DataloaderModule(Dataset):
    def __init__(self, emb_path, train_file, test_file, few_shot = -1, batch_size = 512, num_workers = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.data_list = []
        self.labels = []    
        with open(train_file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    dataPath = os.path.join(emb_path, lineItems[0]+'.npy')
                    label = lineItems[1].split(',')
                    label = [int(i) for i in label]
                    self.data_list.append(dataPath)
                    self.labels.append(label)
                    
        indexes = np.arange(len(self.data_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _data_list= copy.deepcopy(self.data_list)
            _labels= copy.deepcopy(self.labels)
            self.data_list = []
            self.labels = []
            for i in indexes:
                self.data_list.append(_data_list[i])
                self.labels.append(_labels[i])
         
        self.train_set = EmbeddingDataset(self.data_list, self.labels)
        
        self.data_list = []
        self.labels = []    
        with open(test_file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    dataPath = os.path.join(emb_path, lineItems[0]+'.npy')
                    label = lineItems[1].split(',')
                    label = [int(i) for i in label]
                    self.data_list.append(dataPath)
                    self.labels.append(label)
        self.test_set = EmbeddingDataset(self.data_list, self.labels)

        print('#train: ', len(self.train_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)    


class VinDrCXRAll_DataloaderModule(Dataset):
    def __init__(self, emb_path, train_file, test_file, few_shot = -1, batch_size = 512, num_workers = 4):
        super().__init__()
        diseases = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        #diseases = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD']
        #diseases = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.data_list = []
        self.labels = []    
        with open(train_file, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            all_diseases = next(csvReader, None)[2:]
            disease_idxs = [all_diseases.index(d) for d in diseases]
            lines = [line for line in csvReader]
            assert len(lines)/3 == 15000
            for i in range(15000):
                dataPath = os.path.join(emb_path, "train", lines[i*3][0]+".npy")
                label = [0 for _ in range(len(diseases))]
                r1,r2,r3 = lines[i*3][2:],lines[i*3+1][2:],lines[i*3+2][2:] 
                for j, c in enumerate(disease_idxs):
                    label[j] = 1  if int(r1[c])+int(r2[c])+int(r3[c]) > 0 else 0
                self.data_list.append(dataPath)
                self.labels.append(label)
                    
        indexes = np.arange(len(self.data_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _data_list= copy.deepcopy(self.data_list)
            _labels= copy.deepcopy(self.labels)
            self.data_list = []
            self.labels = []
            for i in indexes:
                self.data_list.append(_data_list[i])
                self.labels.append(_labels[i])
         
        self.train_set = EmbeddingDataset(self.data_list, self.labels)
        
        self.data_list = []
        self.labels = []    
        with open(test_file, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            all_diseases = next(csvReader, None)[1:]
            disease_idxs = [all_diseases.index(d) for d in diseases]
            for line in csvReader:
                dataPath = os.path.join(emb_path, "test", line[0]+".npy")
                label = [0 for _ in range(len(diseases))]
                for j, c in enumerate(disease_idxs):
                    label[j] = int(line[c+1])
                self.data_list.append(dataPath)
                self.labels.append(label)
        self.test_set = EmbeddingDataset(self.data_list, self.labels)
        
        print('#train: ', len(self.train_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)        
    
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, epoch, multiclass, log_writter):

    model.train(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for idx, (embedding, lbl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = embedding.shape[0]
        embedding = embedding.double()#.cuda(non_blocking=True)


        if multiclass:
            lbl = lbl.squeeze(1).long()#.cuda(non_blocking=True)
            outputs = model(embedding)
        else:
            lbl = lbl.double()#.cuda(non_blocking=True)
            outputs = torch.sigmoid(model(embedding))

        loss = criterion(outputs,lbl)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=log_writter)
            sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'lr {lr}\t'
              'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses), file=log_writter)
        log_writter.flush()
    return losses.avg

def test(model, test_loader, log_writter):
    model.eval()
    y_test = torch.FloatTensor()#.cuda()
    p_test = torch.FloatTensor()#.cuda()
    with torch.no_grad():
        for idx, (embedding, lbl) in enumerate(test_loader):
            embedding = embedding.double()#.cuda(non_blocking=True)
            lbl = lbl.double()#.cuda(non_blocking=True)

            outputs = model(embedding)
            outputs = torch.sigmoid(outputs)

            p_test = torch.cat((p_test, outputs.data), 0)
            lbl = lbl.type_as(outputs)
            y_test = torch.cat((y_test, lbl), 0)

    mAUC, auc_scores = meanAUC(y_test.cpu().numpy(), p_test.cpu().numpy())
    mMCC, mcc_scores = meanMCC(y_test.cpu().numpy(), p_test.cpu().numpy())
    mAP, ap_scores = meanAP(y_test.cpu().numpy(), p_test.cpu().numpy())
    mF1, f1_scores = meanF1(y_test.cpu().numpy(), p_test.cpu().numpy())

    print("Evaluation:", file=log_writter)
    print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')), file=log_writter)
    print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')), file=log_writter)
    print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')), file=log_writter)
    print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')), file=log_writter)
    
    return [mAUC, mMCC, mAP, mF1], [auc_scores,mcc_scores,ap_scores,f1_scores]

def main(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "cuda"  
    else: 
        device = "cpu"
    exp_path = "/results/linear_eval/{}/{}/{}".format(args.dataset, args.arch, args.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    results = []
    metric_dict = {"auc": [], "mcc": [], "ap": [], "f1": []}
    for run in range(args.runs):    
        model_path = os.path.join(exp_path, "run_{}".format(run))
        print("Run #{}...".format(run))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logs_path = "/results"
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter = open(os.path.join(logs_path, "log.txt"), 'w')
        
        if args.dataset == "ChestDR":
            data = ChestDR_DataloaderModule(args.emb_path, train_file="/data/ChestDR/fewshot-pool.txt", test_file="/data/ChestDR/test.txt", few_shot = args.few_shot)
        elif args.dataset == "VinDrCXRAll":
            data =  VinDrCXRAll_DataloaderModule(args.emb_path, train_file="/data/VinDrCXR/image_labels_train.csv", test_file="/data/VinDrCXR/image_labels_test.csv", few_shot = args.few_shot)
        model = LinearClassifier(args.emb_dim, args.num_class)
        
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "lars":
            optimizer = LARS(model.parameters(), lr=0.15, weight_decay=0)
        model = model.double()
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
            model = model.cuda()
            cudnn.benchmark = True

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1)

        if args.multiclass:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.BCELoss()
   
        best_test_performance = [-1, -1, -1, -1]
        best_test_epoch = -1
        patience_counter = 0
        
        if args.mode == "train":
            for epoch in range(0, args.epochs):

                print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writter)

                loss_avg = train_one_epoch(model, criterion, optimizer, scheduler, data.train_dataloader(), epoch, args.multiclass, log_writter)
                print('Training loss: {}@Epoch: {}'.format(loss_avg, epoch), file=log_writter)
                log_writter.flush()
                scheduler.step()

                test_performance, _ = test(model, data.test_dataloader(), log_writter)

                if test_performance[0] > best_test_performance[0]:
                    save_file = os.path.join(model_path, 'ckpt.pth')
                    save_model(model, optimizer, log_writter, epoch, save_file)
                    print( "Epoch {:04d}: test performance improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_test_performance[0], test_performance[0], save_file))
                    print( "Epoch {:04d}: test performance improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_test_performance[0], test_performance[0], save_file), file=log_writter)
                    best_test_performance = test_performance
                    best_test_epoch = epoch
                    patience_counter = 0
    #             else:
    #                 patience_counter += 1
    #             if patience_counter > args.patience:
    #                 print("Early Stopping")
    #                 break
                log_writter.flush()

            
        else:
            save_file = os.path.join(model_path, 'ckpt.pth')
            checkpoint = torch.load(save_file)
            model.load_state_dict(checkpoint['model'])
            best_test_epoch = checkpoint['epoch']
            best_test_performance, scores = test(model, data.test_dataloader(), log_writter)
            
            metric_dict["auc"].append(scores[0])
            metric_dict["mcc"].append(scores[1])
            metric_dict["ap"].append(scores[2])
            metric_dict["f1"].append(scores[3])
    
        print("Best performance [mAUC, mMCC, mAP, mF1] is {} at epoch {}".format(best_test_performance, best_test_epoch), file=log_writter)
        print("Run {}: Best performance is {} at epoch {}".format(run, best_test_performance, best_test_epoch))
        results.append(best_test_performance)            
    
    with open(os.path.join(exp_path, "log.txt"), 'a') as results_log:
        results = np.array(results)
        avg_std = []
        for i in range(4):
            avg_std.append("{:.4f}({:.4f})".format(np.average(results[:, i]),np.std(results[:, i])))
        print("Average performance from {} runs: {}".format(args.runs, avg_std))
        print("Average performance from {} runs: {}".format(args.runs, avg_std), file=results_log)
        

if __name__ == '__main__':
    args = get_args_parser()
    main(args)
        