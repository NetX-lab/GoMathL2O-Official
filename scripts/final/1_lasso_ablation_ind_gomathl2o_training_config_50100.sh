# 9
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-Mean-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 1 \
    --train-batch-size 64 \
    --loss-func "mean" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-Mean-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 10
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 1 \
    --train-batch-size 64 \
    --loss-func "weighted_sum" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 11
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-Mean-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 1 \
    --train-batch-size 128 \
    --loss-func "mean" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-Mean-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 12
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 1 \
    --train-batch-size 128 \
    --loss-func "weighted_sum" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 13
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-Mean-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 3 \
    --train-batch-size 64 \
    --loss-func "mean" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-Mean-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 14
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 3 \
    --train-batch-size 64 \
    --loss-func "weighted_sum" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 15
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-Mean-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 3 \
    --train-batch-size 128 \
    --loss-func "mean" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-Mean-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

# 16
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC \
    --unroll-length 50 --optimizer-training-steps 100 \
    --epochs 3 \
    --train-batch-size 128 \
    --loss-func "weighted_sum" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"
