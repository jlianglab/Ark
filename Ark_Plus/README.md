## An Open Foundation Model for Chest Radiography

Due to the large size of the datasets, it is not feasible to upload the data and set up a reproducible run of pretraining and finetuning on CodeOcean. Instead, we provide a guideline for the experimental setup and the necessary running commands for **pretraining** and **finetuning**. 

Reproducible runs for **linear-probing** evaluation for Ark+ and CXR Foundation Model on the ChestDR and VinDr-CXR datasets are set up in the `run_linearprobing.sh` file. The pre-generated embeddings of the two models are stored in `.npy` files and uploaded in the `data/ ` forder.

The scripts for generating predictions using the pretraining Ark+ without further training are available in `Zeroshot/Ark+zeroshot-pred.ipynb`. All pre-generated prediction values are stored in CSV files in the `zeroshot_pred_csv/` folder. The AUROC curve for **zero-shot** prediction performance is plotted in `Zeroshot.ipynb`.



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
Modify <PATH_TO_DATASET> in [datasets_config.yaml](./datasets_config.yaml) for each dataset.

(To incorporate a new dataset, refer to the examples provided in datasets_config.yaml. Afterwards, create a corresponding dataloader for the dataset in [dataloader.py](./dataloader.py).)

### Train an Ark model
```
# Train Ark+ with six public datasets
python main_ark.py --data_set MIMIC --data_set CheXpert --data_set ChestXray14 --data_set RSNAPneumonia --data_set VinDrCXR --data_set Shenzhen --opt sgd --warmup-epochs 20  --lr 0.3 --batch_size 50 --model swin_large_768 --init imagenet  --pretrain_epochs 200  --test_epoch 10 --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376  --img_resize 896 --input_size 768

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


