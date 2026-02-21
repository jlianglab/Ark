import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_omni_model, save_checkpoint
from utils import metric_AUROC, cosine_scheduler, DiceLoss, JointLoss
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch, test, evaluate,locked_train_one_epoch
#import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn

# import wandb

sys.setrecursionlimit(40000)

def omni_engine(args, model_path, output_path, source_list, client_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    device = torch.device(args.device)
    cudnn.benchmark = True

    source_list = [str(i) for i in source_list]
    # logs
    exp = 'FedArkPlus_' + args.dataset
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

    # num_classes_list = [len(datasets_config[args.dataset]['diseases'])] #[2]
    # print("num_classes_list:", num_classes_list)


    # training setups
    teachers = [build_omni_model(args) for _ in range(len(client_list))]
    student = build_omni_model(args)
    master = build_omni_model(args)  
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
    print(f"Models are built: they are both {args.model_name} network.")

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
    criterion =  JointLoss()

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
                    print("Training at client #{}, on data from: {}...".format(c, source_list[i]))
                    #locked_train_one_epoch(student, 0, source_list[i], data_loader_list_train[i], device, criterion, optimizer, epoch, args.ema_mode, teachers[c], momentum_schedule, it)
                    for _ in range(4):
                        train_one_epoch(student, c, source_list[i], data_loader_list_train[i], device, criterion, optimizer, epoch, args.ema_mode, teachers[c], momentum_schedule, it)

                for i in client_ds:
                    val_loss = evaluate(teachers[c], c, data_loader_list_val[i], device, criterion, source_list[i])
                    # wandb.log({"client(t)_val_loss_{}".format(dataset_list[i]): val_loss})    

                # Averaging the model parameters
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
                val_loss = evaluate(master, 0, dv, device, criterion, source_list[i])
                val_loss_list.append(val_loss)
                # wandb.log({"val_loss_{}".format(source_list[i]): val_loss})
            
            avg_val_loss = np.average(val_loss_list)
            if args.val_loss_metric == "average":
                val_loss_metric = avg_val_loss
            else:
                val_loss_metric = val_loss_list[source_list.index(args.val_loss_metric)]
            lr_scheduler.step(val_loss_metric)
            
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
                log.write("     Datasets  : " + str(source_list) + "\n")
                log.write("     Val Losses: " + str(val_loss_list) + "\n")
                log.close()
  

            if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                # save_checkpoint(model_save_dict,  filename=save_model_path+str(epoch))
                with open(output_file, 'a') as writer:
                    writer.write("Federated Ark+ Pretraining:\n")
                    writer.write("Epoch {:04d}:\n".format(epoch))
                    t_res, t_res_teacher = [],[]
                    dataset = args.dataset
                    writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                    # diseases = datasets_config[dataset]['diseases']
                    # print(">>{} Disease = {}".format(dataset, diseases))
                    # writer.write("{} Disease = {}\n".format(dataset, diseases))
                    for c, client_ds in enumerate(client_list):
                        for i in client_ds:
                            # multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                            dice_score = test(master,c, data_loader_list_test[i], device)
                            dice_score_teacher = test(teachers[c],c, data_loader_list_test[i], device)
                        # if multiclass:
                            # acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                            # acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                            print(">>{}:Central Master DICE = {} \nLocal Teacher DICE = {}\n".format(source_list[c], dice_score, dice_score_teacher))
                            writer.write(
                                "\n{}:Central Master DICE = {} \nLocal Teacher DICE = {}\n".format(source_list[c], np.array2string(np.array(dice_score), precision=4, separator='\t'), np.array2string(np.array(dice_score_teacher), precision=4, separator='\t')))   
                            t_res.append(dice_score)
                            t_res_teacher.append(dice_score_teacher)
                        
                    # mean_over_all_classes = np.array(t_res).mean()
                    # mean_over_all_classes_teacher = np.array(t_res_teacher).mean()
                    # print(">>Mean: Central Master mACC = {:.4f}, Local Teacher mACC = {:.4f}".format(mean_over_all_classes,mean_over_all_classes_teacher))
                    # writer.write("Mean: Central Master mACC = {:.4f}, Local Teacher mACC = {:.4f}\n".format(mean_over_all_classes,mean_over_all_classes_teacher))
                    
                    # individual_results = metric_AUROC(y_test, p_test, len(diseases))
                    # individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases)) 
                    # print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                    # writer.write(
                    #     "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                    # mean_over_all_classes = np.array(individual_results).mean()
                    # mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                    # print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                    # writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                    # t_res.append(mean_over_all_classes)
                    # t_res_teacher.append(mean_over_all_classes_teacher)
                    
                writer.close()

                test_results.append(t_res)
                test_results_teacher.append(t_res_teacher)
    
        print("\nCentral Master DICE = \n{} \n Local Teacher DICE = \n{}\n".format(test_results, test_results_teacher))
        with open(output_file, 'a') as writer:
            writer.write("\nCentral Master DICE = \n{} \nLocal Teacher DICE = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
        writer.close()

    
        
