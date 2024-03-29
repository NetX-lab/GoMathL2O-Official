# ******************** Standard ********************

python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --optimizer "GOMathL2OSTD" \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapStd-BC-B128 \
    --B-step-size "B" --C-step-size "C" \
    --train-batch-size 128 \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --optimizer "GOMathL2OSTD" \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapStd-BC-B128 \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "B" --C-step-size "C" \
    --device "cuda:0"

# ******************** Standard End ********************



# ******************** Eliminate longer horizong ********************

python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --optimizer "GOMathL2OLH" \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLH-BC-B128 \
    --B-step-size "B" --C-step-size "C" \
    --train-batch-size 128 \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --optimizer "GOMathL2OLH" \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLH-BC-B128 \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "B" --C-step-size "C" \
    --device "cuda:0"

# ******************** Eliminate longer horizong End ********************



# # ******************** Eliminate longer horizong No R ********************

python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLHNoR-BC-B128 \
    --B-step-size "B" --C-step-size "C" \
    --train-batch-size 128 \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLHNoR-BC-B128 \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "B" --C-step-size "C" \
    --device "cuda:0"

# # ******************** Eliminate longer horizong No R End ********************