## A Fully Open, AI Foundation Model Using Heterogeneous Labels Applied to Chest Radiography

## Dataset
1. [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
2. [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
3. [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
4. [VinDrCXR](https://vindr.ai/datasets/cxr)
5. [Shenzhen](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)
6. [MIMIC](https://physionet.org/content/mimic-cxr/2.0.0/)
7. [COVIDx](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2)

## Code
### Requirements
+ Python
+ PyTorch ([pytorch.org](http://pytorch.org))
### Setup environment 
Create and activate a Python 3 conda environment:
```
$ conda create -n ark python=3
$ conda activate ark
```
Install PyTorch according to the [CUDA version](https://pytorch.org/get-started/previous-versions/) (e.g., CUDA 11.6)
```
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

$ cd pretraining/
$ pip install -r requirements
```

### Setup dataset path
Modify <PATH_TO_DATASET> in [datasets_config.yaml](./Pretraining/datasets_config.yaml) for each dataset.

(To incorporate a new dataset, refer to the examples provided in datasets_config.yaml. Afterwards, create a corresponding dataloader for the dataset in [dataloader.py](./Pretraining/dataloader.py).)

### Train an Ark+ model
```
# Train Ark+ with six public datasets
python main_ark.py --data_set MIMIC --data_set CheXpert --data_set ChestXray14 --data_set RSNAPneumonia --data_set VinDrCXR --data_set Shenzhen --opt sgd --warmup-epochs 20  --lr 0.3 --batch_size 50 --model swin_large_768 --init imagenet  --pretrain_epochs 200  --test_epoch 10 --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376  --img_resize 896 --input_size 768

# Train Ark+6 Swin Base version
python main_ark.py --data_set MIMIC --data_set CheXpert --data_set ChestXray14 --data_set RSNAPneumonia --data_set VinDrCXR --data_set Shenzhen --opt sgd --warmup-epochs 20  --lr 0.3  --batch_size 200 --model swin_base --init imagenet  --pretrain_epochs 200  --test_epoch 10 --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376  

# Train Ark+6 ConvNeXt Base version

python main_ark.py --data_set MIMIC --data_set CheXpert --data_set ChestXray14 --data_set RSNAPneumonia --data_set VinDrCXR --data_set Shenzhen --opt sgd --warmup-epochs 20  --lr 0.3 --batch_size 200 --model conv_base --init imagenet  --pretrain_epochs 200 --test_epoch 10  --pretrained_weights https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth --momentum_teacher 0.9  --projector_features 1376  --exp_name projector_1376

```

### Continual training on COVIDx
```
python main_ark.py --data_set MIMIC --data_set CheXpert --data_set ChestXray14 --data_set RSNAPneumonia --data_set VinDrCXR --data_set Shenzhen --data_set COVIDx--opt sgd --warmup-epochs 20  --lr 0.1 --batch_size 50 --model swin_large_768 --init ark  --pretrain_epochs 200  --test_epoch 10 
--pretrained_weights <PRETRAINED_ARK_MODEL>
--momentum_teacher 0.9  --projector_features 1376  --img_resize 896 --input_size 768
```


### Finetune the model on target tasks

```
cd Finetuning/

python main_classification.py --data_set ChestXray14 
--data_dir [PATH_TO_DATASET] 
--train_list dataset/ChestXray14/Xray14_train_official.txt --val_list dataset/ChestXray14/Xray14_val_official.txt --test_list dataset/ChestXray14/Xray14_test_official.txt 
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 
--model swin_large_384 --init ark_plus --key teacher --img_size 896 --input_size 768 --scale_up True
--pretrained_weights [PATH_TO_ARK_MODEL]
```

### Simulate the distributed pretraining across multiple clients

```
cd Distributed/

# If you have Modified <PATH_TO_DATASET> in datasets_config.yaml
cp ../Pretraining/dataloader.py .

# Ark+5 Swin-Base ver. (5-client distribution)
python main_ark_dist.py --opt sgd  --pretrain_epochs 400 --warmup-epochs 20  --lr 0.3 --batch_size 200 --client CheXpert --client ChestXray14 --client RSNAPneumonia --client VinDrCXR --client Shenzhen  --model swin_base --init imagenet --val_loss_metric average  --test_epoch 10  --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376  --exp_name Ark+Base_5clients


# Ark+ Swin-Large ver. (3-client distribution)
python main_ark_dist.py --opt sgd  --pretrain_epochs 50 --warmup-epochs 20  --lr 0.3 --batch_size 50 --client MIMIC --client CheXpert,RSNAPneumonia --client ChestXray14,VinDrCXR,Shenzhen  --model swin_large_768 --init imagenet --val_loss_metric average  --test_epoch 10  --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376  --img_resize 896 --input_size 768 --exp_name Ark+Large_3clients

```