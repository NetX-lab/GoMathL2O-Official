python main.py --config ./configs/1_lasso_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O \
    --device "cuda:0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LASSO-GO-Math-L2O \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand \
    --device "cuda:0"

sh scripts/final/3_lasso_ood_gomathl2o.sh