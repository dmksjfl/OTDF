# Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint

**Authors:** [Jiafei Lyu](https://dmksjfl.github.io/), Mengbei Yan, [Zhongjian Qiao](https://scholar.google.com/citations?user=rFU2fJQAAAAJ&hl=en&oi=ao), [Runze Liu](https://ryanliu112.github.io/), [Xiaoteng Ma](https://xtma.github.io/), [Deheng Ye](https://scholar.google.com/citations?user=jz5XKuQAAAAJ&hl=en&oi=ao), Jingwen Yang, [Zongqing Lu](https://z0ngqing.github.io/), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en)

This is the codes for our ICLR 2025 paper, Optimal Transport Data Filtering (OTDF)

## Method Overview

<img src="https://github.com/dmksjfl/OTDF/blob/master/otdf.png" alt="image" width="600">

## How to run

To run this repo, you do not need to call `pip install -e .`. We run our experiments with Pytorch 1.8 and Gym version 0.23.1. Other dependencies include: `jax==0.4.9, jaxlib==0.4.9, ott-jax==0.4.5, jaxopt==0.8.3`.

To reproduce our reported results in the submission, please check the following instructions:

**Step 1: Solve OT**

One has to first solve the OT problem with `make_otdf_cost.py` for every possible combination of source domain dataset and target domain dataset. For example, one can call the following command to get the solved deviations for medium source domain dataset and medium target domain dataset:

```
CUDA_VISIBLE_DEVICES=0 python make_otdf_cost.py --env halfcheetah --srctype medium --tartype medium
```

This would produce a `hdf5` file in the `costlogs` directory. Note that this is mandatory before running OTDF since it relies on the derived deviations for data filtering and weighting, otherwise an error would occur.

**Step 2: Run OTDF**

After Step 1, we can run OTDF by calling

```
CUDA_VISIBLE_DEVICES=0 python train_otdf.py --env halfcheetah-morph --policy OTDF --srctype medium --tartype medium --weight --reg_weight 0.5 --seed 1
```

## Key Flags

For OTDF, one can specify how many source domain data to keep by `--proportion`, specify whether to include the weights on the source domain data by `--weight` (no weight flag indicates that source domain data will be equally treated), specify the dataset quality of the source domain by `--srctype`, and the dataset quality of the target domain by `--tartype`. One can determine the policy coefficient by specifying `reg_weight`. An example of running OTDF can be found below, and please see more details in `run.sh`,

```
# Example of running OTDF on morph task
CUDA_VISIBLE_DEVICES=0 python train_otdf.py 
    # environment name, e.g., halfcheetah / halfcheetah-morph / halfcheetah-gravity
    --env halfcheetah-morph 
    # policy
    --policy OTDF 
    # source domain dataset quality
    --srctype medium 
    # target domain dataset quality
    --tartype medium 
    # whether to add weight on source domain data
    --weight 
    # how many source domain data to keep
    --proportion 0.8
    # policy coefficient
    --reg_weight 0.5 
    # seed
    --seed 1
```
