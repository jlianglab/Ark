
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_omni_model, save_checkpoint
from utils import metric_AUROC, cosine_scheduler
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch, test_classification, evaluate
#import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn

# import wandb

sys.setrecursionlimit(40000)

def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    device = torch.device(args.device)
    cudnn.benchmark = True

    # logs
    exp = 'Ark_Plus'
    for dataset in dataset_list:
        exp += '_' + dataset 
    model_path = os.path.join(model_path, exp)
    model_path = os.path.join(model_path, args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_file = os.path.join(model_path, "train.log")
    output_file = os.path.join(output_path, exp+"_"+args.exp_name+"_results.txt")

    # dataloaders for pretraining
    data_loader_list_train = []
    for d in dataset_train_list:
        data_loader_list_train.append(DataLoader(dataset=d, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=True))
    data_loader_list_val = []
    for dv in dataset_val_list:
        data_loader_list_val.append(DataLoader(dataset=dv, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=True))
    data_loader_list_test = []
    for dt in dataset_test_list: 
        data_loader_list_test.append(DataLoader(dataset=dt, batch_size=int(args.batch_size/2), shuffle=False,
                                        num_workers=args.workers, pin_memory=True))

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    print("num_classes_list:", num_classes_list)


    # training setups
    model = build_omni_model(args, num_classes_list)
    teacher = build_omni_model(args, num_classes_list)     
    print(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)
    model.to(device)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model_name} network.")

    # momentum parameter is increased to 1. during training with a cosine schedule
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, len(dataset_list))
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += len(d)
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, iters_per_epoch)
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    save_model_path = os.path.join(model_path, exp)

    if args.mode == "train":
        if args.resume:
            resume = save_model_path + '.pth.tar'
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                state_dict = checkpoint['state_dict']
                teacher_state_dict = checkpoint['teacher']
                
                if args.reinit_heads: 
                    for k in model.state_dict().keys():
                        if k.startswith('omni_heads.'):
                            print(f"Removing key {k} from pretrained checkpoint")
                            del state_dict[k]


                model.load_state_dict(state_dict, strict=True)
                teacher.load_state_dict(teacher_state_dict, strict=True)
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss))
                start_epoch += 1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        
            # wandb.init(
            #     # set the wandb project where this run will be logged
            #     project=exp+'_'+args.exp_name,
            #     resume=True
            # )
        # else:
        #     # start a new wandb run to track this script
        #     wandb.init(
        #         # set the wandb project where this run will be logged
        #         project=exp+'_'+args.exp_name,
                
        #         # track hyperparameters and run metadata
        #         config={
        #         "learning_rate": args.lr,
        #         "architecture": args.model_name,
        #         "dataset": exp,
        #         "epochs": args.pretrain_epochs,
        #         }
        #     )

        with open(log_file, 'a') as log:
                log.write(str(args))
        log.close()

        test_results,test_results_teacher = [],[]
        it = start_epoch * len(dataset_list)
        
        for epoch in range(start_epoch, args.pretrain_epochs):
            for i, data_loader in enumerate(data_loader_list_train): 
                criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                train_one_epoch(model, i, dataset_list[i], data_loader, device, criterion, optimizer, epoch, args.ema_mode, teacher, momentum_schedule, it)
                it += 1
            val_loss_list = []
            for i, dv in enumerate(data_loader_list_val):
                criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                val_loss = evaluate(model, i, dv, device, criterion, dataset_list[i])
                val_loss_list.append(val_loss)
                # wandb.log({"val_loss_{}".format(dataset_list[i]): val_loss})
            
            avg_val_loss = np.average(val_loss_list)
            if args.val_loss_metric == "average":
                val_loss_metric = avg_val_loss
            else:
                val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
            lr_scheduler.step(val_loss_metric)
            
            # wandb.log({"avg_val_loss": avg_val_loss})
            
            print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))
            save_checkpoint({
                    'epoch': epoch,
                    'lossMIN': val_loss_list,
                    'state_dict': model.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    },  filename=save_model_path)

            with open(log_file, 'a') as log:
                log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
                log.write("     Datasets  : " + str(dataset_list) + "\n")
                log.write("     Val Losses: " + str(val_loss_list) + "\n")
                log.close()
  

            if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                save_checkpoint({
                     'epoch': epoch,
                     'lossMIN': val_loss_list,
                     'state_dict': model.state_dict(),
                     'teacher': teacher.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': lr_scheduler.state_dict(),
                     },  filename=save_model_path+str(epoch))
                with open(output_file, 'a') as writer:
                    writer.write("Omni-pretraining stage:\n")
                    writer.write("Epoch {:04d}:\n".format(epoch))
                    t_res, t_res_teacher = [],[]
                    for i, dataset in enumerate(dataset_list):
                        writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                        diseases = datasets_config[dataset]['diseases']
                        print(">>{} Disease = {}".format(dataset, diseases))
                        writer.write("{} Disease = {}\n".format(dataset, diseases))

                        multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                        y_test, p_test = test_classification(model, i, data_loader_list_test[i], device, multiclass)
                        y_test_teacher, p_test_teacher = test_classification(teacher, i, data_loader_list_test[i], device, multiclass)
                        if multiclass:
                            acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                            acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                            print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc, acc_teacher))
                            writer.write(
                                "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))   
                            t_res.append(acc)
                            t_res_teacher.append(acc_teacher)

                        if dataset == "CheXpert":
                            test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                            test_diseases = [diseases.index(c) for c in test_diseases_name]
                            y_test = copy.deepcopy(y_test[:,test_diseases])
                            p_test = copy.deepcopy(p_test[:, test_diseases])
                            individual_results = metric_AUROC(y_test, p_test, len(test_diseases)) 
                            y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                            p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                            individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
                        else: 
                            individual_results = metric_AUROC(y_test, p_test, len(diseases))
                            individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases)) 
                        print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                        writer.write(
                            "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                        mean_over_all_classes = np.array(individual_results).mean()
                        mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                        print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                        writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                        t_res.append(mean_over_all_classes)
                        t_res_teacher.append(mean_over_all_classes_teacher)
                        
                    writer.close()

                    test_results.append(t_res)
                    test_results_teacher.append(t_res_teacher)
        
                    print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
        with open(output_file, 'a') as writer:
            writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
        writer.close()

    
        
