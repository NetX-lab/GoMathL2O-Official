
# Before running these commands, please read and run "optimizees/logistic_real_data.py"

# [generating test instances and the solutions and saving them]

# solve problems with FISTA and save the optimal objective value for each instance (5,000 iterations are sufficient to obtain optimal objective)
# python main.py --config ./configs/logistic_testing.yaml --optimizer ProximalGradientDescentMomentum \
#     --input-dim 34 --output-dim 4601 --rho 1e-1 \
#     --test-size 1 --test-batch-size 1 --test-length 5000 \
#     --load-mat --save-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --save-dir LogisticL1-FISTA --loss-save-path test_losses_ionosphere


# # [test our method L2O-PA, L2O-DM, L2O-RNNprop and Ada-LISTA]
# python main.py --config ./configs/logistic_testing.yaml --p-use --a-use --save-dir LogisticL1-L2O-PA \
#     --input-dim 34 --output-dim 4601 --rho 1e-1 \
#     --test-size 1 --test-batch-size 1 --test-length 5000 \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --loss-save-path test_losses_ionosphere --device "cuda:0"

# python main.py --config ./configs/logistic_testing.yaml --optimizer CoordBlackboxLSTM --grad-method bp_grad --save-dir LogisticL1-L2O-DM \
#     --input-dim 34 --output-dim 4601 --rho 1e-1 \
#     --test-size 1 --test-batch-size 1 --test-length 5000 \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --loss-save-path test_losses_ionosphere --device "cuda:0"

# python main.py --config ./configs/logistic_testing.yaml --optimizer RNNprop --grad-method bp_grad --save-dir LogisticL1-L2O-RNNprop \
#     --input-dim 34 --output-dim 4601 --rho 1e-1 \
#     --test-size 1 --test-batch-size 1 --test-length 5000 \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --loss-save-path test_losses_ionosphere --device "cuda:0"

# python main_unroll.py --config ./configs/logistic_testing.yaml --load-mat --load-sol \
#     --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LogisticL1-AdaLISTA \
#     --input-dim 34 --output-dim 4601 --rho 1e-1 \
#     --test-size 1 --test-batch-size 1 --test-length 5000 \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --loss-save-path test_losses_ionosphere --device "cuda:0" \
#     --input-dim 34 --test-batch-size 1 --test-size 1


# # [test other hand-designed optimizers]
# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol \
#     --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --optimizer ProximalGradientDescentMomentum \
#     --save-dir LogisticL1-FISTA \
#     --loss-save-path test_losses_ionosphere --device "cuda:0" \
#     --input-dim 34 --test-batch-size 1 --test-size 1 --output-dim 4601

# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol \
#     --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --optimizer ProximalGradientDescent \
#     --save-dir LogisticL1-ISTA \
#     --loss-save-path test_losses_ionosphere --device "cuda:0" \
#     --input-dim 34 --test-batch-size 1 --test-size 1 --output-dim 4601

# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol \
#     --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --optimizer Adam --step-size 1e-2 --momentum1 1e-1 --momentum2 1e-1 \
#     --save-dir LogisticL1-Adam --loss-save-path test_losses_ionosphere --device "cuda:0" \
#     --input-dim 34 --test-batch-size 1 --test-size 1 --output-dim 4601

# python main.py --config ./configs/logistic_testing.yaml --load-mat --load-sol \
#     --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
#     --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 \
#     --hyper-step 1e-07 --save-dir LogisticL1-AdamHD --loss-save-path test_losses_ionosphere \
#     --device "cuda:0" \
#     --input-dim 34 --test-batch-size 1 --test-size 1 --output-dim 4601


# Our proposed
python main.py --config ./configs/logistic_testing_wo_xk_set.yaml --r-use --q-use --b-use  \
    --save-dir LogisticL1-GO-Math-L2O \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-ionosphere-rho1e-1 \
    --loss-save-path test_losses_ionosphere --test-length 5000 \
    --device "cuda:0" \
    --input-dim 34 --test-batch-size 1 --test-size 1 --output-dim 4601