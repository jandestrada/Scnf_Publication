import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from harness.th_model_classes.class_sklearn_classification import SklearnClassification
from harness.th_model_classes.class_sklearn_regression import SklearnRegression

"""
rocklins_features = ['avg_all_frags', 'net_atr_net_sol_per_res', 'n_charged', 'buried_np_afilmvwy_per_res',
                     'avg_best_frag', 'fa_atr_per_res', 'exposed_polars', 'unsat_hbond',
                     'mismatch_probability', 'hbond_lr_bb', 'exposed_np_afilmvwy', 'fa_rep_per_res',
                     'degree', 'p_aa_pp', 'netcharge', 'worstfrag', 'frac_sheet', 'buried_np_per_res',
                     'abego_res_profile_penalty', 'hbond_sc', 'holes', 'cavity_volume', 'score_per_res',
                     'hydrophobicity', 'hbond_bb_sc', 'ss_sc', 'contig_not_hp_max', 'contact_all', 'omega',
                     'exposed_hydrophobics', 'contig_not_hp_avg']

rocklins_EHEE_features = rocklins_features + ['abd50_mean', 'abd50_min', 'dsc50_mean', 'dsc50_min',
                                              'ssc50_mean', 'ssc50_min']
rocklins_cols_to_use_per_topology_dict = {'HHH': rocklins_features, 'EHEE': rocklins_EHEE_features,
                                          'HEEH': rocklins_features, 'EEHEE': rocklins_features}
"""


def rocklins_logistic_classifier():
    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1, n_jobs=-1)
    th_model = SklearnClassification(model=rocklins_logistic_model, model_author='Hamed',
                                     model_description="Rocklin Logistic: penalty='l1' and C=0.1")
    return th_model


def rocklins_linear_regression():
    rocklins_linear_model = LinearRegression()
    th_model = SklearnRegression(model=rocklins_linear_model, model_author='Hamed',
                                 model_description='Rocklin LinReg: Default sklearn linear regression')
    return th_model


def rocklins_gradboost_regression():
    rocklins_gradboost_model = GradientBoostingRegressor(n_estimators=250, max_depth=5, min_samples_split=5, learning_rate=0.01, loss='ls')
    th_model = SklearnRegression(model=rocklins_gradboost_model, model_author='Hamed',
                                 model_description="Rocklin GradientBoostingRegressor: n_estimators=250, max_depth=5, min_samples_split=5, learning_rate=0.01, and loss='ls'")
    return th_model
