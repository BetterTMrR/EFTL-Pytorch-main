# Enhancing Semi-Supervised Domain Adaptation via Effective Target Labeling


This is a Pytorch implementation of AAAI 2024 paper "Enhancing Semi-Supervised Domain Adaptation via Effective Target Labeling".

Our code is based on [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME), [SHOT](https://github.com/tim-learn/SHOT), and [NRC](https://github.com/Albert0147/SFDA_neighbors) implementations.
We thank their awesome works.


## Install

`pip install -r requirements.txt`


## Data Preparation
Please download [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/), [Office-Home](http://ai.bu.edu/visda-2017/), and [DomainNet](http://ai.bu.edu/M3SDA/) to ./data

## Start Training

To run training on A to C task on dataset Office-Home,


(1) Pre-train the source model,

`python train_source.py --dset office_home --s 0 --net resnet34 --max_epoch 20
`

(2) Adaptation under 1-shot and 3-shot settings using FixMME as baseline,


`python main.py --dset office_home --s 0 --t 1 --shot 3 --net resnet34 --use_src --method FixMME --th 0.85
`

`python main.py --dset office_home --s 0 --t 1 --shot 1 --net resnet34 --use_src --method FixMME --th 0.85 
`

(3) Adaptation under 1-shot and 3-shot settings using MME as baseline,

`python main.py --dset office_home --s 0 --t 1 --shot 3 --net resnet34 --use_src --method MME 
`

`python main.py --dset office_home --s 0 --t 1 --shot 1 --net resnet34 --use_src --method MME 
`
