python main.py --config ./configs/logistic_training_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LogisticL1-GO-Math-L2O \
    --device "cuda:1"

python main.py --config ./configs/logistic_testing_wo_xk_set.yaml \
    --r-use --q-use --b-use \
    --save-dir LogisticL1-GO-Math-L2O \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/logistic-rand \
    --loss-save-path losses-rand \
    --device "cuda:1"

sh scripts/final/logistic_ood_gomathl2o.sh