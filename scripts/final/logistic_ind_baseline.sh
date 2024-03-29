
# [generating test instances and the solutions and saving them]

## generate L1-regularized Logistic Regression instances and save them to "./optimizees/matdata/logistic-rand"
# python main.py --config ./configs/logistic_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA --save-to-mat --optimizee-dir ./optimizees/matdata/logistic-rand --device "cuda:0"

## solve L1-regularized Logistic Regression with FISTA and save the optimal objective value for each instance (5,000 iterations are sufficient to obtain optimal objective)
# python main.py --config ./configs/logistic_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA --load-mat --save-sol --optimizee-dir ./optimizees/matdata/logistic-rand --test-length 5000 --device "cuda:0"


# [train models for out method, L2O-DM and L2O-RNNprop]
python main.py --config ./configs/logistic_training.yaml --p-use --a-use --save-dir LogisticL1-L2O-PA --device "cuda:0"
# python main.py --config ./configs/logistic_training.yaml --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LogisticL1-L2O-DM --device "cuda:0"
# python main.py --config ./configs/logistic_training.yaml --optimizer RNNprop --grad-method bp_grad --save-dir LogisticL1-L2O-RNNprop --device "cuda:0"


# [test L2O-DM and L2O-RNNprop]
python main.py --config ./configs/logistic_testing.yaml --p-use --a-use --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --save-dir LogisticL1-L2O-PA --device "cuda:0"
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LogisticL1-L2O-DM --device "cuda:0"
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer RNNprop --grad-method bp_grad --save-dir LogisticL1-L2O-RNNprop --device "cuda:0"


# # [train and test models for Ada-LISTA] //This may take long time for problems with size of 250*500.
# python main_unroll.py --optimizer AdaLISTA --optimizee-type LogisticL1 --input-dim 50 --sparsity 20 --output-dim 1000 --layers 10 --init-lr 2e-3 --save-dir LogisticL1-AdaLISTA --device "cuda:0"
# python main_unroll.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LogisticL1-AdaLISTA --device "cuda:0" --test-batch-size 4


# [test other hand-designed optimizers]
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer ProximalGradientDescentMomentum --save-dir LogisticL1-FISTA --device "cuda:0"
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer ProximalGradientDescent --save-dir LogisticL1-ISTA --device "cuda:0"
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer Adam --step-size 1e-2 --momentum1 1e-1 --momentum2 1e-1 --save-dir LogisticL1-Adam --device "cuda:0"
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 --hyper-step 1e-07 --save-dir LogisticL1-AdamHD --device "cuda:0"


# Test L2O PA OOD
sh scripts/final/logistic_ood_baseline_l2opa.sh