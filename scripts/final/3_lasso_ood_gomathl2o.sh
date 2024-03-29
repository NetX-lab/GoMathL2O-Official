

# **************************** S only ood start ****************************

# S+: 10 20 50 100
python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S10 --test-length 1000 --ood --ood-s 10.0 --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S20 --test-length 1000 --ood --ood-s 20.0 --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S50 --test-length 1000 --ood --ood-s 50.0 --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S100 --test-length 1000 --ood --ood-s 100.0 --ood-t 0.0

# S-: -10 -20 -50 -100
python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S-10 --test-length 1000 --ood --ood-s "-10.0" --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S-20 --test-length 1000 --ood --ood-s "-20.0" --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S-50 --test-length 1000 --ood --ood-s "-50.0" --ood-t 0.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-S-100 --test-length 1000 --ood --ood-s "-100.0" --ood-t 0.0

# **************************** S only ood end ****************************


# **************************** T only ood start ****************************
# T+: 10 20 50 100
python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T10 --test-length 1000 --ood --ood-s 0.0 --ood-t 10.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T20 --test-length 1000 --ood --ood-s 0.0 --ood-t 20.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T50 --test-length 1000 --ood --ood-s 0.0 --ood-t 50.0

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T100 --test-length 1000 --ood --ood-s 0.0 --ood-t 100.0

# T-: -10 -20 -50 -100
python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T-10 --test-length 1000 --ood --ood-s 0.0 --ood-t "-10.0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T-20 --test-length 1000 --ood --ood-s 0.0 --ood-t "-20.0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T-50 --test-length 1000 --ood --ood-s 0.0 --ood-t "-50.0"

python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
    --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
    --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
    --loss-save-path losses-rand-ood-T-100 --test-length 1000 --ood --ood-s 0.0 --ood-t "-100.0"


# **************************** T only ood end ****************************


# # **************************** S+T=0 ood start ****************************
# # : 10 20 50 100
# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S10T-10 --test-length 1000 --ood --ood-s 10.0 --ood-t "-10.0"

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S20T-20 --test-length 1000 --ood --ood-s 20.0 --ood-t "-20.0"

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S50T-50 --test-length 1000 --ood --ood-s 50.0 --ood-t "-50.0"

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S100T-100 --test-length 1000 --ood --ood-s 100.0 --ood-t "-100.0"

# #  -10 -20 -50 -100
# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S-10T10 --test-length 1000 --ood --ood-s "-10.0" --ood-t 10.0

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S-20T20 --test-length 1000 --ood --ood-s "-20.0" --ood-t 20.0

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S-50T50 --test-length 1000 --ood --ood-s "-50.0" --ood-t 50

# python main.py --config ./configs/2_lasso_testing_wo_xk_set.yaml --r-use --q-use --b-use   \
#     --save-dir LASSO-GO-Math-L2O --device "cuda:0" \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --loss-save-path losses-rand-ood-S-100T100 --test-length 1000 --ood --ood-s "-100.0" --ood-t 100.0


# # **************************** S+T=0 ood end ****************************