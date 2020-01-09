# Learning to Self-Train for Semi-Supervised Few-Shot Classification
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

This repository contains the TensorFlow implementation for [NeurIPS 2019](https://nips.cc/) Paper ["Learning to Self-Train for Semi-Supervised Few-Shot Classification"](https://arxiv.org/pdf/1906.00562.pdf)

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Running Experiments](#running-experiments)
* [Acknowledgements](#acknowledgements)

## Installation

In order to run this repository, we advise you to install python 2.7 or 3.5 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install tensorflow on it:

```bash
conda create --name lst-tf python=2.7
conda activate lst-tf
conda install tensorflow-gpu=1.3.0
```

Install other requirements:
```bash
pip install scipy tqdm opencv-python pillow matplotlib
```

Clone this repository:

```bash
git clone https://github.com/xinzheli1217/learning-to-self-train.git 
cd learning-to-self-train
```

## Project Architecture

```
.
├── data_generator              # dataset generator 
|   └── meta_data_generator.py  # data genertor for meta-train phase
├── models                      # tensorflow model files 
|   ├── models.py               # resnet12 CNN class
|   └── meta_model_LST.py       # semi-supervised meta-train model class
├── trainer                     # tensorflow trianer files  
|   └── meta_LST.py             # semi-supervised meta-train trainer class
├── utils                       # a series of tools used in this repo
|   └── misc.py                 # miscellaneous tool functions
| 
├── data                        # the folder containing datasets for experiments
├── pretrain_weights_dir        # the folder containing MTL pre-training weights
├── weights_saving_dir          # the folder containing meta-training weights
├── test_output_dir             # the folder containing meta-testing files
├── filenames_and_labels        # the folder containing image file paths and labels for experiments
|
├── exp_train.py                # the python file with main function and parameter settings for meta-training
└── exp_test.py                 # the python file with main function and parameter settings for meta-testing
```

## Running Experiments

First, download our processed images: miniImagenet[\[Download Page\]](https://drive.google.com/open?id=1ont6qSoBRHdQbTdEei15_ak-FagCej9S) or tieredImagenetp[\[Download Page\]](https://drive.google.com/file/d/17xk0kVDCQOQ5JUFpGIbTCQgzpzl-Gyhu/view?usp=sharing), move the unziped folder to `./data`. And then download the pre-trained models: miniImagenet[\[Download Page\]](https://drive.google.com/open?id=1Qh89u-UYbXsflvx8w5c47j9pfjD-blG8) or tieredImagenet, move the unziped folder to `./pretrain_weights_dir`. 

### Training from Pre-Trained Models
Run semi-supervised meta-train phase (e.g. 𝑚𝑖𝑛𝑖ImageNet, 1-shot) :
```bash
python exp_train.py --shot_num=1 --dataset='miniImagenet' --pretrain_class_num=64 --nb_ul_samples=10 --metatrain_iterations=15000 --exp_name='LST_mini_1_shot'
```
Run semi-supervised meta-test phase (e.g. 𝑚𝑖𝑛𝑖ImageNet, 1-shot) :
```bash
python exp_test.py --shot_num=1 --dataset='miniImagenet' --pretrain_class_num=64 --use_distractors=False --nb_ul_samples=100 --unfiles_num=10 --test_iter=15000 --recurrent_stage_nums=6 --nums_in_folders=30 --hard_selection=20 --exp_name='LST_mini_1_shot' 
```

### Hyperparameters and Options
There are some main hyperparameters used in the experiments, you can edit them in the `exp_train.py` and the `exp_test.py` file for meta-train and meta-test phase respectively. There are two kinds of hyperparameters: (1) common hyperparameters that shared with meta-train and meta-test, (2) test-specific hyperparameters that used for recurrent self-training process in meta-test.
* Common hyperparameters:
  - `way_num` number of classes
  - `shot_num` number of examples per class
  - `dataset` dataset used in the experiment (miniImagenet or tieredImagenet)
  - `pretrain_class_num` number of meta-train classes
  - `exp_name` name for the current experiment
  - `meta_batch_size` number of tasks sampled per meta-update in meta-train phase
  - `base_lr` step size alpha for inner gradient update
  - `meta_lr` the meta learning rate for SS and initial model parameters
  - `min_meta_lr` the min meta learning rate for all meta-parameters
  - `swn_lr` the meta learning rate for SWN
  - `nb_ul_samples` number of unlabeled examples per class
  - `re_train_epoch_num` number of re-training inner gradient updates
  - `train_base_epoch_num` number of total inner gradient updates during train (meta-train only)
  - `test_base_epoch_num` number of total inner gradient updates during test (meta-test only)
  
* Test-specific hyperparameters:
  - `use_distractors` if using distractor classes during meta-test
  - `num_dis` number of distracting classes used for meta-testing
  - `unfiles_num` number of unlabeled sample files used in the experiment (There are 10 unlabeled samples per class in each file)
  - `recurrent_stage_nums` number of recurrent stages used during meta-test
  - `local_update_num` number of inner gradient updates used in each recurrent stage
  - `nums_in_folders` number of unlabeled samples (per class) used in each recurrent stage
  - `hard_selection` number of remaining samples (per class) after applying hard-selection
  
If you want to change other settings, please see the comments and descriptions in `exp_train.py` and `exp_test.py`.

### Performance

|          (%)           | 𝑚𝑖𝑛𝑖  | 𝒕𝒊𝒆𝒓𝒆𝒅  |  𝑚𝑖𝑛𝑖 (w/D) | 𝒕𝒊𝒆𝒓𝒆𝒅  (w/D) |
| ---------------------- | ------------ | ------------ | ------------ | ------------ |
| 1-shot            | `70.1 ± 1.9` | `77.7 ± 1.6` |  `64.1 ± 1.9` | `73.5 ± 1.6` |
| 5-shot           | `78.7 ± 0.8` | `85.2 ± 0.8` |  `77.4 ± 1.8` | `83.4 ± 0.8` |

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{li2019lst,
  title={Learning to Self-Train for Semi-Supervised Few-Shot Classification},
  author={Xinzhe Li and Qianru Sun and Yaoyao Liu and Shibo Zheng and Qin Zhou and Tat{-}Seng Chua and Bernt Schiele},
  booktitle={NeurIPS},
  year={2019}
}
```

## Acknowledgements

Our implementations use the source code from the following repositories and users:
* [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning)
* [Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)
