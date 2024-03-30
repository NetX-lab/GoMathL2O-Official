# Towards Robust Learning to Optimize with Theoretical Guarantees

This repo implements the proposed model in the CVPR 2024 paper: Towards Robust Learning to Optimize with Theoretical Guarantees. 
Based on the Math-L2O framework [2], our theory part formulated the L2O model's convergences in two typical OOD scenarios: Initial point OOD and objective OOD. 
Our modeling part proposes a gradient-only L2O model to reduce the L2O model's input feature magnitude. For the technical details, please find our paper below.

## Introduction
Our work aims to improve robustness in solving minimization problems like:
$$\min_{x}F(x) = f(x) + r(x)$$
where $f$ is a smooth convex function and $r$ is proper, convex and, possibly non-smooth.


**(Optimizee).** 
In folder `./optimizees/`, each instance of $F(x)$ , two types: LASSO and Logistic+ $L_1$-norm. We borrow the implementations in [1] for $f$'s gradient, the proximal operator for $r$, and $F$'s subgradient ($L_1$ -norm's subgradient is calculated by $sign(x)$ ). We implement two boundaries of $r$'s subgradient set, reproducible training and validating data generation with given seeds, and $L$-smoothness calculation for Logistic.

**(Optimizer).** 
In the folder `./optimizers/`, iterative learning-based and non-learning algorithms for solving optimizes. We borrow the following implementations in [1]:
* Hand-designed optimizers: Adam, ISTA, FISTA, Shampoo, and subgradient descent. 
* Adaptive hyperparameter tuning: Adam-HD. 
* Algorithm unrolling: Ada-LISTA. 
* LSTM-based optimizers learned from data: Coordinate Blackbox LSTM, RNNprop, Coordinate Math LSTM. 
* **Our proposed LSTM-based optimizers: Go-Math-L2O**.

**(Training).**
We provide two loss functions, which can be selected by the `loss-func` term in the folder `./configs/.` Mean (set by `mean`):
$$L(\theta) = E_{f,r} \left[\frac{1}{K}\sum_{k=1}^{K} f(x_k) + r(x_k) \right].$$
Weighted-sum (set by `weighted_sum`):
$$L(\theta) = E_{f,r} \left[\sum_{k=1}^{K} \left(k\Big/\left(\sum_{k=1}^{K}k\right)\right) f(x_k) + r(x_k) \right].$$
In both functions, $x_k$ is the solution at $k$-th iteration. $\theta$ represents the parameters in the L2O model, and $K$ denotes the number of iterations. The `unroll-length` term in the folder `./configs/` can configure $K$.



## Software dependencies
Our codes depend on the following packages:
```
cudatoolkit=10.1
pytorch=1.12
configargparse=1.5.3
scipy=1.10.1
```

## Start up: A toy example
We borrow the following toy example configuration in [1].
$$F(x) = \frac{1}{2}\|\|Ax-b\|\|^2_2 + \lambda \|\|x\|\|_1$$

where $A\in\mathbb{R}^{20\times40},x\in\mathbb{R}^{40},b\in\mathbb{R}^{20}$. An optimizer is then trained to find solutions for these generated optimizes.

Training:
```
python main.py --config ./configs/0_lasso_toy.yaml
```
The model will be stored at `./results/LASSO-toy/GOMathL2O.pth`.

Testing:
```
python main.py --config ./configs/0_lasso_toy.yaml --test
```

## Reproduction

### Configurations 
Following [1], all the configurations are listed in `main.py`. 
You can specify values for particular parameters in a `.yaml` file or the Python command. The command's values will **overwrite** those in the YAML file.

To reproduce the results in our paper, you may check YAML files in `./configs/` and run commands in `./scripts/`.

### Reproducing LASSO results in the main pages

To obtain Figure 1, please run `./scripts/2_lasso_ind_baseline.sh` and `2_lasso_ind_gomathl2o.sh`.

To obtain Figure 2, please run `./scripts/3_lasso_ood_real.sh`

To obtain Figures 3 and 4, please run `./scripts/3_lasso_ood_baseline_l2opa.sh` and `3_lasso_ood_gomathl2o.sh`.

### Reproducing ablation results in the appendix

To obtain Figure 5, please run `./scripts/1_lasso_ablation_ind_gomathl2o_grad_map.sh`

To obtain Figures 6, 7, 8, and 9, please run `./scripts/1_lasso_ablation_ind_gomathl2o_training_config_20100.sh`, `./scripts/1_lasso_ablation_ind_gomathl2o_training_config_50100.sh`, and `./scripts/1_lasso_ablation_ind_gomathl2o_training_config_100_100.sh`. This may take a long time.

### Reproducing Logistic results in the appendix
To obtain Figures 10, 11, and 12 please run `./scripts/1_lasso_ablation_ind_gomathl2o_Q_config.sh` and `./scripts/1_lasso_ablation_ood_gomathl2o_Q_config.sh`.

To obtain Figure 13, please run `./scripts/logistic_ind_baseline.sh` and `logistic_ind_gomathl2o.sh`.

To obtain Figures 14 and 15, please run `./scripts/logistic_ood_real_ionosphere.sh` and `logistic_ood_real_spambase.sh`.

To obtain Figures 16 and 17, please run `./scripts/logistic_ood_baseline_l2opa.sh` and `logistic_ood_gomathl2o.sh`.


## Tips

**(Memory).** 24GB GPU memory is enough for all default configurations in our experiments.


## Citing our work
Please cite our paper if you find our codes helpful in your research or work.
```
@inproceedings{song2024gomathl2o,
  title     = {Towards Robust Learning to Optimize with Theoretical Guarantees},
  author    = {Song, Qingyu and Lin, Wei and Wang, Juncheng and Xu, Hong},
  booktitle = {{Proc.~IEEE/CVF CVPR}},
  year      = {2024}
}
```

## Acknowledgement
This repository is developed based on the official implementation [1] of Math-L2O [2].


## References
[1]. Jialin Liu, Xiaohan Chen, HanQin Cai, (2023 Aug 2). MS4L2O. Github. https://github.com/xhchrn/MS4L2O.

[2]. Jialin Liu, Xiaohan Chen, Zhangyang Wang, Wotao Yin, and HanQin Cai. Towards Constituting Mathematical Structures for Learning to Optimize. In *ICML*, 2023.
