# [generating test instances and save them]

## generate LASSO instances and save them to "./optimizees/matdata/lasso-rand"
# python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --save-to-mat --optimizee-dir ./optimizees/matdata/lasso-rand

## solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
# python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --load-mat --save-sol --optimizee-dir ./optimizees/matdata/lasso-rand --test-length 5000



# # [train models for L2O-DM and L2O-RNNprop]
python main.py --config ./configs/1_lasso_training.yaml --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LASSO-L2O-DM --device "cuda:0"
python main.py --config ./configs/1_lasso_training.yaml --optimizer RNNprop --init-lr 3e-3 --val-freq 5 --grad-method bp_grad --save-dir LASSO-L2O-RNNprop --device "cuda:0"

# # [test L2O-DM and L2O-RNNprop]
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LASSO-L2O-DM --device "cuda:0"
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer RNNprop --grad-method bp_grad --save-dir LASSO-L2O-RNNprop --device "cuda:0"

# [train and test models for Ada-LISTA] //This may take long time for problems with size of 250*500.
python main_unroll.py --optimizer AdaLISTA --optimizee-type LASSO --input-dim 500 --sparsity 50 --output-dim 250 --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA --device "cuda:0"
python main_unroll.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA --device "cuda:0" --test-batch-size 4

# [test other hand-designed optimizers]
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --device "cuda:0"
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer ProximalGradientDescent --save-dir LASSO-ISTA --device "cuda:0"
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer Adam --step-size 1e-2 --momentum1 1e-1 --momentum2 1e-1 --save-dir LASSO-Adam --device "cuda:0"
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 --hyper-step 1e-07 --save-dir LASSO-AdamHD --device "cuda:0"

# Train and test L2O PA
python main.py --config ./configs/1_lasso_training.yaml --p-use --a-use --save-dir LASSO-L2O-PA --device "cuda:0"
python main.py --config ./configs/2_lasso_testing.yaml --load-mat --load-sol --p-use --a-use \
    --save-dir LASSO-L2O-PA --optimizee-dir ./optimizees/matdata/lasso-rand --device "cuda:0"

# # Test L2O PA OOD
# sh scripts/final/3_lasso_ood_baseline_l2o_pa.sh