import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int, get_config
from dataloader import  *

import torch
from engine import ark_engine

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="vit_base|vit_small|swin_base|swin_tiny", default="vit_base", type="string")
    parser.add_option("--init", dest="init",
                      help="Random| ImageNet_1k| ImageNet_21k| SAM| DeiT| BEiT| DINO| MoCo_V3| MoBY | MAE| SimMIM",
                      default="Random", type="string")
    parser.add_option("--pretrained_weights", dest="pretrained_weights", help="Path to the Pretrained model", default=None, type="string")
    parser.add_option("--from_checkpoint", dest="from_checkpoint", help="whether load pretrained weights from checkpoint", default=False, action="callback", callback=vararg_callback_bool)
    # parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
    #                   default=14, type="int")
    parser.add_option("--data_set", dest="dataset_list", help="ChestXray14|CheXpert|Shenzhen|VinDrCXR|RSNAPneumonia",  action="append")
    # parser.add_option("--finetune_dataset", dest="finetune_dataset", help="ChestXray14|CheXpert|Shenzhen|VinDrCXR|RSNAPneumonia", default="VinDrCXR", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    # parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
    parser.add_option("--epochs", dest="epochs", help="num of epoches", default=200, type="int")
    parser.add_option("--exp_name", dest="exp_name", default="", type="string")
    
    parser.add_option("--ema_mode", dest="ema_mode", default="epoch", help="update teacher model at which time (epoch | iteration)", type="string")
    parser.add_option('--momentum_teacher', default=0.9, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_option("--pretrain_epochs", dest="pretrain_epochs", help="num of omni-pretraining epoches", default=10, type="int")
    # parser.add_option("--reinit_heads", dest="reinit_heads", help="whether re-initialize the heads from pre-trained model", default=False, action="callback", callback=vararg_callback_bool)
    parser.add_option("--test_epoch", dest="test_epoch", help="whether test after every epoch", default=1, type="int")                
    # parser.add_option("--reuse_head", dest="reuse_head", help="whether reuse the head from pre-trained model", default=False, action="callback",
    #                   callback=vararg_callback_bool)
    # parser.add_option("--copy_head", dest="copy_head", help="copy the weights of head x to head y (e.g., 0-4)", default='', type="string")
    parser.add_option("--val_loss_metric", dest="val_loss_metric", help="which validation loss for early stop and model save (average | [dataset])", default="average", type="string")                  
    parser.add_option("--projector_features", dest="projector_features", help="num of projector features", default=1376, type="int")
    parser.add_option("--use_mlp", dest="use_mlp", help="whether use mlp for projector", default=False, action="callback",
                      callback=vararg_callback_bool)
    # Optimizer parameters
    parser.add_option('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_option('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_option('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_option('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_option('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_option('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_option('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_option('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_option('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_option('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_option('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_option('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_option('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_option('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_option('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_option('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_option('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_option('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')


    # parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
    # parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
    #                   callback=vararg_callback_bool)
    # parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    # parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    # parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
    #                   callback=vararg_callback_bool)
    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")


    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)

    exp_name = args.model_name + "_" + args.init
    model_path = os.path.join("./Models",exp_name)
    output_path = os.path.join("./Outputs",exp_name)

    datasets_config = get_config('datasets_config.yaml')
    for dataset in args.dataset_list:
        assert dataset in list(datasets_config.keys())

    dataset_train_list = []
    dataset_val_list = []
    dataset_test_list = []
    for dataset in args.dataset_list:
        dataset_train_list.append(
            dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['train_list'],
                                 augment=None)
        )
        dataset_val_list.append(
            dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['val_list'],
                                 augment=build_transform_classification(normalize=args.normalization, mode="valid"))
        )
        dataset_test_list.append(
            dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['test_list'],
                                 augment=build_transform_classification(normalize=args.normalization, mode="test"))
        )

    ark_engine(args, model_path, output_path, args.dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list, finetune_dataset)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

