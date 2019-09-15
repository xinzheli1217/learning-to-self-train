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
git clone https://github.com/yaoyao-liu/meta-transfer-learning.git 
cd learning-to-self-train
```
