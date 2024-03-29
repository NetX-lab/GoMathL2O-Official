from results_plot import *

# ********************* LASSO Results *********************


# NOTE 1: Figure 1, InD comparison LASSO
files_1 = ['LASSO-ISTA', 'LASSO-FISTA', 'LASSO-AdaLISTA',
           'LASSO-L2O-DM', 'LASSO-L2O-RNNprop',
           'LASSO-L2O-PA', 'LASSO-Adam', 'LASSO-AdamHD',
           'LASSO-GO-Math-L2O']

legends_1 = ['ISTA', 'FISTA', 'AdaLISTA',
             'L2O-DM', 'L2O-RNNprop',
             'L2O-PA', 'Adam', 'AdamHD',
             'GO-Math-L2O']

name = "final/figure1_ind_cmp_lasso"

plot_opt_processes(files_1, None, legends_1, None, name,
                   y_ticks=[1e2, 1e-2, 1e-4, 1e-6, 1e-7])


# NOTE 2: Figure 2, Real-World comparison LASSO
files_1 = ['LASSO-ISTA', 'LASSO-FISTA',
           'LASSO-L2O-DM', 'LASSO-L2O-RNNprop',
           'LASSO-L2O-PA', 'LASSO-Adam', 'LASSO-AdamHD',
           'LASSO-GO-Math-L2O']
#
legends_1 = ['ISTA', 'FISTA',
             'L2O-DM', 'L2O-RNNprop',
             'L2O-PA', 'Adam', 'AdamHD',
             'GO-Math-L2O']

name = "final/figure2_ood_cmp_lasso_real"


plot_opt_processes_real(files_1, None, legends_1, None, name,
                        data_file="losses-real", y_ticks=[1e2, 1e-2, 1e-4, 1e-6, 5e-8])


# NOTE 3: Figure 3 OOD S shift LASSO
losses_files = ['losses-rand-ood-S10',
                'losses-rand-ood-S20',
                'losses-rand-ood-S50',
                'losses-rand-ood-S100',
                'losses-rand-ood-S-10',
                'losses-rand-ood-S-20',
                'losses-rand-ood-S-50',
                'losses-rand-ood-S-100']
legends = ['$s=10$',
           '$s=20$',
           '$s=50$',
           '$s=100$',
           '$s=-10$',
           '$s=-20$',
           '$s=-50$',
           '$s=-100$']

files_1 = ['LASSO-GO-Math-L2O/' +
           f for f in losses_files]
files_2 = ['LASSO-L2O-PA/' + f for f in losses_files]

legends_1 = ['GO-Math-L2O, ' + f for f in legends]
legends_2 = ['L2O-PA, ' + f for f in legends]

name = "final/figure3_ood_cmp_lasso_s"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, [1e6, 1e-2, 1e-4, 1e-6, 1e-7], _loc='lower left')


# NOTE 4: Figure 4 OOD T shift LASSO
losses_files = ['losses-rand-ood-T10',
                'losses-rand-ood-T20',
                'losses-rand-ood-T50',
                'losses-rand-ood-T100',
                'losses-rand-ood-T-10',
                'losses-rand-ood-T-20',
                'losses-rand-ood-T-50',
                'losses-rand-ood-T-100']
legends = ['$t=10$',
           '$t=20$',
           '$t=50$',
           '$t=100$',
           '$t=-10$',
           '$t=-20$',
           '$t=-50$',
           '$t=-100$']

files_1 = ['LASSO-GO-Math-L2O/' +
           f for f in losses_files]
files_2 = ['LASSO-L2O-PA/' + f for f in losses_files]

legends_1 = ['GO-Math-L2O, ' + f for f in legends]
legends_2 = ['L2O-PA, ' + f for f in legends]

name = "final/figure4_ood_cmp_lasso_t"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, [1e5, 1e4, 1e3, 1e2, 1e1, 1e0])


# NOTE 5: Figure 5, InD comparison Gradient Map Ablation Study
files_1 = ['LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapStd-BC-B128',
           'LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLH-BC-B128',
           'LASSO-GO-Math-L2O/ablation_grad_map/RQB-GradMapLHNoR-BC-B128']

legends_1 = ['STD', 'LH', 'LHNoR']

name = "final/figure5_grad_map_ablation_lasso"

plot_opt_processes(files_1, None, legends_1, None, name)


# NOTE 6: Figure 6, Training Configuration LASSO, 20/100 .
files_1 = ['LASSO-GO-Math-L2O/ablation_train_config/100_20-1epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-1epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-1epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-1epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC']

legends_1 = ['1', '2', '3', '4', '5', '6', '7', '8']

name = "final/figure6_training_config_cmp_100_20_lasso"

plot_training_config_opt_processes(files_1, legends_1, name)


# NOTE 7: Figure 7, Training Configuration LASSO, 50/100.
files_1 = ['LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-1epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC']

legends_1 = ['9', '10', '11', '12', '13', '14', '15', '16']

name = "final/figure7_training_config_cmp_100_50_lasso"

plot_training_config_opt_processes(files_1, legends_1, name)


# NOTE 8: Figure 8, Training Configuration LASSO, 100/100.
files_1 = ['LASSO-GO-Math-L2O/ablation_train_config/100_100-1epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-1epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-1epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-1epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-3epoch-B64-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-3epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC']

legends_1 = ['17', '18', '19', '20', '21', '22', '23', '24']

name = "final/figur8_training_config_cmp_100_100_lasso"

plot_training_config_opt_processes(files_1, legends_1, name)


# NOTE 9: Figure 9, Training Configuration LASSO, best of all.
files_1 = ['LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B128-Mean-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_20-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B64-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_50-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC',
           'LASSO-GO-Math-L2O/ablation_train_config/100_100-3epoch-B128-WeightedSum-RQB-GradMapLHNoR-BC']
legends_1 = ['7', '8', '14', '16', '24']

name = "final/figure9_training_config_cmp_best_lasso"

plot_training_config_opt_processes(files_1, legends_1, name)


# NOTE 10: Figure 10, Q config InD LASSO.
files_1 = ['LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-Q',
           'LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QsqrtL',
           'LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QL',
           'LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QLL']
legends_1 = ['$Q$', '$Q/\sqrt{L}$', '$Q/L$', '$Q/{L^2}$']

name = "final/figure10_q_config_cmp_B128"

plot_training_config_opt_processes(files_1, legends_1, name)


# NOTE 11: Figure 11, Q config OOD LASSO S shifting.
losses_files = ['losses-rand-ood-S10',
                'losses-rand-ood-S20',
                'losses-rand-ood-S50',
                'losses-rand-ood-S100',
                'losses-rand-ood-S-10',
                'losses-rand-ood-S-20',
                'losses-rand-ood-S-50',
                'losses-rand-ood-S-100']
legends = ['$s=10$',
           '$s=20$',
           '$s=50$',
           '$s=100$',
           '$s=-10$',
           '$s=-20$',
           '$s=-50$',
           '$s=-100$']

files_1 = ['LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-Q/' +
           f for f in losses_files]
files_2 = ['LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QsqrtL/' +
           f for f in losses_files]

legends_1 = ['$Q$, ' + f for f in legends]
legends_2 = ['$Q/\sqrt{L}$, ' + f for f in legends]

name = "final/figure11_q_config_ood_cmp_lasso_s"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, [1e6, 1e-2, 1e-4, 1e-6, 1e-7], _loc='lower left')


# NOTE 12: Figure 12, Q config OOD T shift LASSO
losses_files = ['losses-rand-ood-T10',
                'losses-rand-ood-T20',
                'losses-rand-ood-T50',
                'losses-rand-ood-T100',
                'losses-rand-ood-T-10',
                'losses-rand-ood-T-20',
                'losses-rand-ood-T-50',
                'losses-rand-ood-T-100']
legends = ['$t=10$',
           '$t=20$',
           '$t=50$',
           '$t=100$',
           '$t=-10$',
           '$t=-20$',
           '$t=-50$',
           '$t=-100$']

files_1 = ['LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-Q/' +
           f for f in losses_files]
files_2 = ['LASSO-GO-Math-L2O/ablation_q_config/RQB-GradMapLHNoR-QsqrtL/' +
           f for f in losses_files]

legends_1 = ['$Q$, ' + f for f in legends]
legends_2 = ['$Q/\sqrt{L}$, ' + f for f in legends]

name = "final/figure12_q_config_ood_cmp_lasso_t"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, [1e5, 1e4, 1e3, 1e2, 1e1, 1e0])


# ********************* LASSO Results End *********************


# ********************* Logistic Results *********************

# NOTE 13: Figure 13, InD comparison Logistic
files_1 = ['LogisticL1-ISTA', 'LogisticL1-FISTA', 'LogisticL1-AdaLISTA',
           'LogisticL1-L2O-DM', 'LogisticL1-L2O-RNNprop',
           'LogisticL1-L2O-PA',
           'LogisticL1-Adam',
           'LogisticL1-AdamHD',
           'LogisticL1-GO-Math-L2O'
           ]

legends_1 = ['ISTA', 'FISTA', 'AdaLISTA',
             'L2O-DM', 'L2O-RNNprop',
             'L2O-PA',
             'Adam',
             'AdamHD',
             'GO-Math-L2O'
             ]

name = "final/figure13_ind_cmp_logisticl1"

plot_opt_processes(files_1, None, legends_1, None, name,
                   y_ticks=[1e2, 1e-2, 1e-4, 1e-6, 1e-8])


# NOTE: OOD Logistic real data
files_1 = ['LogisticL1-ISTA', 'LogisticL1-FISTA',
           'LogisticL1-L2O-DM', 'LogisticL1-L2O-RNNprop', 'LogisticL1-L2O-PA',
           'LogisticL1-Adam', 'LogisticL1-AdamHD', 'LogisticL1-GO-Math-L2O']

legends_1 = ['ISTA', 'FISTA',
             'L2O-DM', 'L2O-RNNprop', 'L2O-PA',
             'Adam', 'AdamHD', 'GO-Math-L2O']


# NOTE 14: Figure 14 OOD Logistic real data: Ionoshpere
name = "final/figure14_ood_cmp_logistic_ionoshpere"
plot_opt_processes(files_1, None, legends_1, None, name,
                   data_file="test_losses_ionosphere")

# NOTE 15: Figure 15 OOD Logistic real data: Spambase
name = "final/figure15_ood_cmp_logistic_spambase"
plot_opt_processes(files_1, None, legends_1, None, name,
                   data_file="test_losses_spambase")


# NOTE 16: Figure 16 OOD S shift
losses_files = ['losses-rand-ood-S10',
                'losses-rand-ood-S20',
                'losses-rand-ood-S50',
                'losses-rand-ood-S100',
                'losses-rand-ood-S-10',
                'losses-rand-ood-S-20',
                'losses-rand-ood-S-50',
                'losses-rand-ood-S-100']
legends = ['$s=10$',
           '$s=20$',
           '$s=50$',
           '$s=100$',
           '$s=-10$',
           '$s=-20$',
           '$s=-50$',
           '$s=-100$']

files_1 = ['LogisticL1-GO-Math-L2O/' + f for f in losses_files]
files_2 = ['LogisticL1-L2O-PA/' + f for f in losses_files]

legends_1 = ['GO-Math-L2O, ' + f for f in legends]
legends_2 = ['L2O-PA, ' + f for f in legends]

name = "final/figure16_ood_cmp_logistic_s"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, _loc='lower left',
    y_ticks=[1e2, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-7])


# NOTE 17: Figure 17 OOD T shift
losses_files = ['losses-rand-ood-T10',
                'losses-rand-ood-T20',
                'losses-rand-ood-T50',
                'losses-rand-ood-T100',
                'losses-rand-ood-T-10',
                'losses-rand-ood-T-20',
                'losses-rand-ood-T-50',
                'losses-rand-ood-T-100']
legends = ['$t=10$',
           '$t=20$',
           '$t=50$',
           '$t=100$',
           '$t=-10$',
           '$t=-20$',
           '$t=-50$',
           '$t=-100$']

files_1 = ['LogisticL1-GO-Math-L2O/' + f for f in losses_files]
files_2 = ['LogisticL1-L2O-PA/' + f for f in losses_files]

legends_1 = ['GO-Math-L2O, ' + f for f in legends]
legends_2 = ['L2O-PA, ' + f for f in legends]

name = "final/figure17_ood_cmp_logistic_t"
plot_ablation_ood_processes(
    files_1, files_2, legends_1, legends_2, name, [1e2, 1e1, 1e0, 1e-1, 1e-2])


# ********************* Logistic Results End *********************
