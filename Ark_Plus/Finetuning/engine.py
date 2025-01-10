
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
import csv

from models import build_classification_model, save_checkpoint
from utils import *
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from trainer import train_one_epoch, evaluate, test_classification, test_model

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

sys.setrecursionlimit(40000)


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  data_loader_test = DataLoader(dataset=dataset_test, batch_size=int(args.batch_size/2), shuffle=False,
                            num_workers=args.workers, pin_memory=True)  
  # training phase
  if args.mode == "train":
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
                           
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....")
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      criterion = torch.nn.BCEWithLogitsLoss()
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        criterion = torch.nn.CrossEntropyLoss()
        print("use CrossEntropyLoss...")
      model = build_classification_model(args)
      print(model)

      if args.freeze_encoder:
        print("===> freezing encoder...")
        ##freeze all layers but the head 
        for name, param in model.named_parameters():
          if name not in ['head.weight', 'head.bias']:
              param.requires_grad = False

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      #optimizer = torch.optim.Adam(parameters, lr=args.lr)
      # optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0, momentum=args.momentum, nesterov=False)
      # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
      #                                  threshold=0.0001, min_lr=0, verbose=True)
      optimizer = create_optimizer(args, model)
      loss_scaler = NativeScaler()

      lr_scheduler, _ = create_scheduler(args, optimizer)

      if args.resume:
        resume = os.path.join(model_path, experiment + '.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          best_val_loss = init_loss
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, init_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))

      mean_result_list, result_list = [], []
      accuracy = []
      for epoch in range(start_epoch, args.epochs):
        if args.skip_training:
          break
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)

        val_loss = evaluate(data_loader_val, device,model, criterion)

        lr_scheduler.step(val_loss)

        if args.test_every_epoch:
          y_test, p_test = test_model(model, data_loader_test, args)
          y_test = y_test.cpu().numpy()
          p_test = p_test.cpu().numpy()

          if args.data_set in ["RSNAPneumonia", "COVIDx"]:
            acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
            print(">>{}: ACCURACY = {}".format(experiment,acc))
            with open(output_file, 'a') as writer:
              writer.write(
                "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
            accuracy.append(acc)
          
          if test_diseases is not None:
            y_test = copy.deepcopy(y_test[:,test_diseases])
            p_test = copy.deepcopy(p_test[:,test_diseases])

          mAUC, auc_scores = meanAUC(y_test, p_test)
          mMCC, mcc_scores = meanMCC(y_test, p_test)
          mAP, ap_scores = meanAP(y_test, p_test)
          mF1, f1_scores = meanF1(y_test, p_test)
            
          print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
          print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
          print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
          print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
          mean_result_list.append(mAUC)
          result_list.append([mAUC,mMCC,mAP,mF1])
        
        if val_loss < best_val_loss:
          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                              save_model_path))
          best_val_loss = val_loss
          patience_counter = 0  
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
          },  filename="{}".format(save_model_path, epoch))

        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
          patience_counter += 1

        if patience_counter >= args.patience:
          print("Early Stopping")
          break
          
      #save_checkpoint({
      #      'state_dict': model.state_dict(),
      #    },  filename="{}({} epoch)".format(save_model_path, epoch))
  
      # log experiment
      with open(log_file, 'a') as f:
        f.write("{} ({}epoch)\n".format(experiment,epoch-args.patience))
        f.close()

      if len(mean_result_list) > 0:
        best_rest = max(mean_result_list)
        best_epoch = mean_result_list.index(best_rest)
        print("=====> Max result:  {} at epoch {}".format(best_rest, best_epoch) )
        print("mAUC, mMCC, mAP, mF1 = {}".format(result_list[best_epoch]))
        with open(output_file, 'a') as writer:
          writer.write("=====> Max result:  {} at epoch {}\n".format(best_rest, best_epoch))
          writer.write("mAUC, mMCC, mAP, mF1 = {}\n".format(result_list[best_epoch]))
 
  print ("start testing.....")


  log_file = os.path.join(model_path, "models.log")
  if not os.path.isfile(log_file):
    print("log_file ({}) not exists!".format(log_file))
  else:
    accuracy = []
    mean_auc, mean_mcc, mean_ap, mean_f1 = [],[],[],[]
    metric_dict = {"auc": [], "mcc": [], "ap": [], "f1": []}
    with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
      experiment = reader.readline()
      print(">> Disease = {}".format(diseases))
      writer.write("Disease = {}\n".format(diseases))

      while experiment:
        experiment = experiment.split()[0]
        saved_model = os.path.join(model_path, experiment + ".pth.tar")
        pred_csv = os.path.join(model_path, experiment + ".csv")
        gt_csv = os.path.join(model_path, "gt.csv")
        if os.path.exists(pred_csv) and os.path.exists(gt_csv):
          y_test = read_from_csv(gt_csv)
          p_test = read_from_csv(pred_csv)
        else:
          y_test, p_test = test_classification(saved_model, data_loader_test, device, args)
          y_test = y_test.cpu().numpy()
          p_test = p_test.cpu().numpy()

        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
          print(">>{}: ACCURACY = {}".format(experiment,acc))
          writer.write(
            "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
          accuracy.append(acc)

        
        if test_diseases is not None:
          y_test = copy.deepcopy(y_test[:,test_diseases])
          p_test = copy.deepcopy(p_test[:,test_diseases])
        
        mAUC, auc_scores = meanAUC(y_test, p_test)
        mMCC, mcc_scores = meanMCC(y_test, p_test)
        mAP, ap_scores = meanAP(y_test, p_test)
        mF1, f1_scores = meanF1(y_test, p_test)
          
        print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
        print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
        print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
        print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
        writer.write("AUC = {}\n MCC = {}\nAP = {}\nF1 = {}\n".format(np.array2string(np.array(auc_scores), precision=4, separator=','), 
                                                                    np.array2string(np.array(mcc_scores), precision=4, separator=','), 
                                                                    np.array2string(np.array(f1_scores), precision=4, separator=','),
                                                                    np.array2string(np.array(auc_scores), precision=4, separator=',')))
        writer.write("{}: mAUC = {:.4f}, mMCC = {:.4f}, mAP = {:.4f}, mF1 = {:.4f}\n".format(experiment, mAUC, mMCC, mAP, mF1))
        
        
        
        data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
        data = data + p_test.tolist()
        print(len(data[0]),len(data[1]))
        # Write data to CSV file
        with open(pred_csv, mode='w', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerows(data)

        mean_auc.append(mAUC)
        mean_mcc.append(mMCC)
        mean_ap.append(mAP)
        mean_f1.append(mF1)
        metric_dict["auc"].append(auc_scores)
        metric_dict["mcc"].append(mcc_scores)
        metric_dict["ap"].append(ap_scores)
        metric_dict["f1"].append(f1_scores)
        experiment = reader.readline()
    
      
      data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
      data = data + y_test.tolist()
      print(len(data[0]),len(data[1]))
      # Write data to CSV file
      with open(gt_csv, mode='w', newline='') as file:
          csvwriter = csv.writer(file)
          csvwriter.writerows(data)

      mean_auc,mean_mcc,mean_ap,mean_f1 = np.array(mean_auc),np.array(mean_mcc),np.array(mean_ap),np.array(mean_f1)
      print(">> All trials: mAUC = {}\n mMCC = {}\n mAP = {}\n mF1 = {}\n ".format(np.array2string(mean_auc, precision=4, separator=','),
                                                                                   np.array2string(mean_mcc, precision=4, separator=','),
                                                                                   np.array2string(mean_ap, precision=4, separator=','),
                                                                                   np.array2string(mean_f1, precision=4, separator=',')))
      writer.write("All trials: mAUC  = {}\n mMCC  = {}\n mAP = {}\n mF1 = {}\n ".format(np.array2string(mean_auc, precision=4, separator='\t'),
                                                                            np.array2string(mean_mcc, precision=4, separator='\t'),
                                                                            np.array2string(mean_ap, precision=4, separator='\t'),
                                                                            np.array2string(mean_f1, precision=4, separator='\t')))
      print(">> Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                            np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                            np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                            np.mean(mean_f1), np.std(mean_f1)))
      writer.write("Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                                np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                                np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                                np.mean(mean_f1), np.std(mean_f1)))
      class_wise_scores = []
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["auc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["mcc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["ap"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["f1"]))
      writer.write("AUC/MCC/AP/F1(mean/std): \n{}\n".format(class_wise_scores))

      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        accuracy = np.array(accuracy)
        print(">> All trials: ACCURACY  = {}".format(np.array2string(accuracy, precision=4, separator=',')))
        writer.write("All trials: ACCURACY  = {}\n".format(np.array2string(accuracy, precision=4, separator='\t')))
      