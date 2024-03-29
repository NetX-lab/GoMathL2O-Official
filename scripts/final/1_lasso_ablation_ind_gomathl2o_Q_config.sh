
# 1. Q 
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-Q \
    --B-step-size "B" --C-step-size "C" \
    --device "cuda:0" 

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-Q \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "B" --C-step-size "C" \
    --device "cuda:0"

# 2. QsqrtL
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QsqrtL \
    --B-step-size "BsqrtL" --C-step-size "C" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QsqrtL \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "BsqrtL" --C-step-size "C" \
    --device "cuda:0"

# 3. QL
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QL \
    --B-step-size "BL" --C-step-size "C" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QL \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "BL" --C-step-size "C" \
    --device "cuda:0"

# 4. QLL
python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QLL \
    --B-step-size "BLL" --C-step-size "C" \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QLL \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --B-step-size "BLL" --C-step-size "C" \
    --device "cuda:0"