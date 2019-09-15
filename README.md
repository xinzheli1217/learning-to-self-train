# Learning to Self-Train for Semi-Supervised Few-Shot Classification TensorFlow
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Running Experiments](#running-experiments)

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
├── exp_test.py                 # the python file with main function and parameter settings for meta-testing
└── run_experiment.sh           # the script to run the whole experiment
```

## Running Experiments

First, download processed images: miniImagenet[\[Download Page\]] or tieredImagenet[\[Download Page\]], move the unziped folder to `./data`. And then download the pre-trained models: miniImagenet[\[Download Page\]] or tieredImagenet[\[Download Page\]], move the unziped folder to `./pretrain_weights_dir`. 

### Semi-Supervised Meta-Training (e.g. 𝑚𝑖𝑛𝑖ImageNet, 1-shot)

```bash
python exp_train.py --shot_num=1 --dataset='miniImagenet' --nb_ul_samples=10 --metatrain_iterations=15000
```
