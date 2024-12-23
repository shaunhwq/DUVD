# Notes

Installation instructions (3.8 works too)
```
conda create -n DUVD python=3.8
conda activate DUVD
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install matplotlib opencv-python
```

# Author's

Depth-aware Unpaired Video Dehaizng
===============================================
This is the PyTorch implementation of the paper 'Depth-aware Unpaired Video Dehaizng'.

Prerequisites
---------------------------------
* Python 3.7
* Pytorch
* NVIDIA GPU + CUDA cuDNN

The detailed prerequiesites are in `environment.yml`

Datasets
---------------------------------
### 1.Data for training and testing.
After downloading the dataset, please use scripts/flist.py to generate the file lists. For example, to generate the file list on the revide testset, you should run:

```
python scripts/flist.py --path path_to_REVIDE_hazy_path --output ./datasets/revide_test.flist
```

Please notice that we conduct experiments on 4x downsampled version of the REVIDE dataset, you should downsample it first.
[REVIDE dataset](https://github.com/BookerDeWitt/REVIDE_Dataset) | [NYU-Depth](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html) | [Collected Real-world videos](https://drive.google.com/file/d/16_p7n7FO36Hm2-hvZkNe1xVq1cWK8Kc0/view?usp=drive_link)


Getting Started
--------------------------------------
To use the pre-trained models, download it from the following link then copy it to the corresponding checkpoints folder. For instance, if you want to test the model on nyu/real-world hazy frames, download the pretrained model for nyu/real-world and put it under  `./checkpoints/test_real`. If you use the light model, remember to change the BASE_CHANNEL_NUM in config.yml from 96 to 64.

[Pretrained model on REVIDE](https://drive.google.com/file/d/1E1E_4oK7e1YTYOd3WzQ9wI7PWAVp5M1O/view?usp=drive_link) | [Pretrained model on NYU-depth/Real-world](https://drive.google.com/file/d/1gF6PBdCHSSq6jkkeLGB5Ag0oMGOTJRyN/view?usp=drive_link) | [Pretrained light model on REVIDE](https://drive.google.com/file/d/1qwe5ZjQSQzo-QoiDETAj7fDohSLvFKGk/view?usp=drive_link) | [Pretrained light model on NYU-Depth/Real-world](https://drive.google.com/file/d/1F2ywh0YuAYGHmS13Hvx1bm1GHqFSUWwd/view?usp=drive_link)

### 1. Training 
1) Prepare the training datasets following the operations in the Datasets part. 
2) Add a config file 'config.yml' in the checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/train_example`. Make sure TRAIN_CLEAN_FLIST and TRAIN_HAZY_FLIST are right. 
3) Train the model, for example:
```
python train.py --model 1 --checkpoints ./checkpoints/train_example
```

### 2. Testing
1)Prepare the testing datasets following the operations in the Datasets part.
2)Put the trained weight in the checkpoint folder 
3)Add a config file 'config.yml' in your checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/`, 
4)Test the model, for example:
```
python test.py --model 1 --checkpoints ./checkpoints/test_revide
```
For quick testing, you can download the checkpoint on real-world frames and put it to the corresponding folder `./checkpoints/test_real` and run test on our example frames directly using

```
python test.py --model 1 --checkpoints ./checkpoints/test_real
```
