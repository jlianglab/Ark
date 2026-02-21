
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_omni_model, build_omni_model_from_checkpoint, save_checkpoint
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

def omni_engine(args, model_path, output_path, dataset_list, client_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    device = torch.device(args.device)
    cudnn.benchmark = True

    # logs
    exp = 'Distributed_Ark'
    for client_ds in args.client_list:
        exp += '_' + client_ds 
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
                                        num_workers=int(args.workers/2), pin_memory=True))

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    print("num_classes_list:", num_classes_list)
   

    # training setups
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.from_checkpoint:
        teachers = [build_omni_model_from_checkpoint(args, num_classes_list, 'teacher'+str(i)) for i in range(len(client_list))]
        student = build_omni_model_from_checkpoint(args, num_classes_list, 'master')  
        master = build_omni_model_from_checkpoint(args, num_classes_list, 'master')  
    else:
        teachers = [build_omni_model(args, num_classes_list) for _ in range(len(client_list))]
        student = build_omni_model(args, num_classes_list)
        master = build_omni_model(args, num_classes_list)
    print(student)

    if torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student)
        teachers = [torch.nn.DataParallel(teacher) for teacher in teachers]
        master = torch.nn.DataParallel(master)
    
    student.to(device)
    for teacher in teachers:
        teacher.to(device) 
        for p in teacher.parameters():
            p.requires_grad = False
    
    master.to(device)
    for p in master.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model_name} network.")

    # momentum parameter is increased to 1. during training with a cosine schedule
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, 1)
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += len(d)
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, iters_per_epoch)

    optimizer = create_optimizer(args, student)
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
                master_state_dict = checkpoint['master']
                master.load_state_dict(master_state_dict, strict=True)
                for i, teacher in enumerate(teachers):
                    teacher_state_dict = checkpoint['teacher'+str(i)]
                    teacher.load_state_dict(teacher_state_dict, strict=True)
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss))
                start_epoch += 1
            elif os.path.isfile(args.pretrained_weights):
                print("=> loading checkpoint from '{}'".format(args.pretrained_weights))
                checkpoint = torch.load(args.pretrained_weights)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                master_state_dict = checkpoint['master']
                master.load_state_dict(master_state_dict, strict=False)
                for i, teacher in enumerate(teachers):
                    tname = 'teacher'+str(i)
                    if tname not in checkpoint.keys():
                        print("Loading master for {}".format(tname))
                        teacher.load_state_dict(checkpoint['master'], strict=False)
                    else:
                        teacher_state_dict = checkpoint[tname]
                        teacher.load_state_dict(teacher_state_dict, strict=False)
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
            # start a new wandb run to track this script
            # wandb.init(
            #     # set the wandb project where this run will be logged
            #     project=exp+'_'+args.exp_name,
                
            #     # track hyperparameters and run metadata
            #     config={
            #     "learning_rate": args.lr,
            #     "architecture": args.model_name,
            #     "dataset": exp,
            #     "epochs": args.pretrain_epochs,
            #     }
            # )

        if not os.path.exists(save_model_path+ '.pth.tar'):
            save_checkpoint({
                    'master': master.state_dict(),
                    },  filename=save_model_path)

        with open(log_file, 'a') as log:
                log.write(str(args))
        log.close()

        test_results,test_results_teacher = [],[]
        it = start_epoch 
        for epoch in range(start_epoch, args.pretrain_epochs):

            for p in master.parameters():
                p.zero_()

            for c, client_ds in enumerate(client_list):
                state_dict = torch.load(save_model_path+ '.pth.tar')['master'] 
                student.load_state_dict(state_dict, strict=True)

                for i in client_ds:
                    print("Training at client #{}, on dataset: {}...".format(c, dataset_list[i]))
                    train_one_epoch(student, i, dataset_list[i], data_loader_list_train[i], device, criterion, optimizer, epoch, args.ema_mode, teachers[c], momentum_schedule, it)

                for i in client_ds:
                    val_loss = evaluate(teachers[c], i, data_loader_list_val[i], device, criterion, dataset_list[i])
                    # wandb.log({"client(t)_val_loss_{}".format(dataset_list[i]): val_loss})    

                with torch.no_grad():
                    for (name, param_q), param_k in zip(student.named_parameters(), master.parameters()):
                        if "omni_heads" not in name:
                            param_k.data.add_((1.0/len(client_list)) * param_q.detach().data)
                        for i in client_ds:
                            if "omni_heads.{}".format(i) in name:
                                #print(name, param_k.detach().data)
                                param_k.data.add_(param_q.detach().data)
            it += 1

            val_loss_list = []
            for i, dv in enumerate(data_loader_list_val):
                val_loss = evaluate(master, i, dv, device, criterion, dataset_list[i])
                val_loss_list.append(val_loss)
                # wandb.log({"server(m)_val_loss_{}".format(dataset_list[i]): val_loss})
            
            avg_val_loss = np.average(val_loss_list)
            if args.val_loss_metric == "average":
                val_loss_metric = avg_val_loss
            else:
                val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
            lr_scheduler.step(val_loss_metric)

            # log metrics to wandb
            # wandb.log({"avg_val_loss": avg_val_loss})

            print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))
            model_save_dict = {
                    'epoch': epoch,
                    'lossMIN': val_loss_list,
                    'master': master.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    }
            for i, teacher in enumerate(teachers):
                model_save_dict['teacher'+str(i)] = teacher.state_dict()
            save_checkpoint(model_save_dict,  filename=save_model_path)

            with open(log_file, 'a') as log:
                log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
                log.write("     Datasets  : " + str(dataset_list) + "\n")
                log.write("     Val Losses: " + str(val_loss_list) + "\n")
                log.close()

            if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                save_checkpoint(model_save_dict,  filename=save_model_path+str(epoch))

                with open(output_file, 'a') as writer:
                    writer.write("Omni-pretraining stage:\n")
                    writer.write("Epoch {:04d}:\n".format(epoch))
                    t_res, t_res_teacher = [],[]
                    for c, client_ds in enumerate(client_list):
                        for i in client_ds:
                            print("Testing at client #{}, on dataset: {}...".format(c, dataset_list[i]))
                            writer.write("{} Validation Loss = {:.5f}:\n".format(dataset_list[i], val_loss_list[i]))
                            diseases = datasets_config[dataset_list[i]]['diseases']
                            print(">>{} Disease = {}".format(dataset_list[i], diseases))
                            writer.write("{} Disease = {}\n".format(dataset_list[i], diseases))

                            multiclass =  datasets_config[dataset_list[i]]['task_type'] == "multi-class classification"
                            y_test, p_test = test_classification(master, i, data_loader_list_test[i], device, multiclass)
                            y_test_teacher, p_test_teacher = test_classification(teachers[c], i, data_loader_list_test[i], device, multiclass)
                            if multiclass:
                                acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                                acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                                print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset_list[i],acc, acc_teacher))
                                writer.write(
                                    "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset_list[i], np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))   
                                t_res.append(acc)
                                t_res_teacher.append(acc_teacher)

                            if dataset_list[i] == "CheXpert":
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
                            print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset_list[i], np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            writer.write(
                                "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset_list[i], np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            mean_over_all_classes = np.array(individual_results).mean()
                            mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                            print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset_list[i], mean_over_all_classes,mean_over_all_classes_teacher))
                            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset_list[i], mean_over_all_classes,mean_over_all_classes_teacher))
                            t_res.append(mean_over_all_classes)
                            t_res_teacher.append(mean_over_all_classes_teacher)
                        
                    writer.close()

                    test_results.append(t_res)
                    test_results_teacher.append(t_res_teacher)
        
                    print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
        with open(output_file, 'a') as writer:
            writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
        writer.close()


    else:
        # training one more epoch at client end
        t_res, t_res_teacher = [],[]
        checkpoint = torch.load(save_model_path+ '.pth.tar')
        epoch = checkpoint['epoch']
        # state_dict = checkpoint['master'] # distribute student or teacher
        master_state_dict = checkpoint['master']
        if torch.cuda.device_count() == 1:
            master_state_dict = {k.replace('module.', ''): v for k, v in master_state_dict.items() if k.startswith('module.')}
        master.load_state_dict(master_state_dict, strict=True)
        for i, teacher in enumerate(teachers):
            teacher_state_dict = checkpoint['teacher'+str(i)]
            if torch.cuda.device_count() == 1:
                teacher_state_dict = {k.replace('module.', ''): v for k, v in teacher_state_dict.items() if k.startswith('module.')}
            teacher.load_state_dict(teacher_state_dict, strict=True)
        print("=> loaded checkpoint '{}' (epoch={:04d})".format(save_model_path, epoch))
        for c, client_ds in enumerate(client_list):
        #     student.load_state_dict(state_dict, strict=True)

            for i in client_ds:
                print("Training at client #{}, on dataset: {}...".format(c, dataset_list[i]))
        #         train_one_epoch(student, i, dataset_list[i], data_loader_list_train[i], device, criterion, optimizer, epoch, args.ema_mode, teachers[c], momentum_schedule, 1)
        
            
                dataset = dataset_list[i]
                diseases = datasets_config[dataset]['diseases']
                print(">>{} Disease = {}".format(dataset, diseases))
    
                multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                y_test, p_test = test_classification(master, i, data_loader_list_test[i], device, multiclass)
                y_test_teacher, p_test_teacher = test_classification(teachers[c], i, data_loader_list_test[i], device, multiclass)
                if multiclass:
                    acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                    acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                    print(">>{}:Master ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset_list[i],acc, acc_teacher))
                    t_res.append(acc)
                    t_res_teacher.append(acc_teacher)
    
                if dataset == "CheXpert":
                    test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                    test_diseases = [diseases.index(c) for c in test_diseases_name]
                    y_test = copy.deepcopy(y_test[:, test_diseases])
                    p_test = copy.deepcopy(p_test[:, test_diseases])
                    individual_results = metric_AUROC(y_test, p_test, len(test_diseases)) 
                    y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                    p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                    individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
                else: 
                    individual_results = metric_AUROC(y_test, p_test, len(diseases))
                    individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases)) 
                print(">>{}:Master AUC = {}, \nTeacher AUC = {}\n".format(dataset_list[i], np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                # print(">>{}:Client model's AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t')))
                
                mean_over_all_classes = np.array(individual_results).mean()
                mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                print(">>{}: Master mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset_list[i], mean_over_all_classes,mean_over_all_classes_teacher))
                t_res.append(mean_over_all_classes)
                t_res_teacher.append(mean_over_all_classes_teacher)

        print(t_res)
        print(t_res_teacher)

