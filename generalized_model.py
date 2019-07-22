import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
    

def main():
    # Reading in data from versioned-datasets repo.
    file_path = '/Users/janestrada/Desktop/SorelleSummerResearch/TestHarness/dataframes/df_aggregated_spc.csv'
    combined_data = pd.read_csv(file_path)
    combined_data['dataset_original'] = combined_data['dataset']
    
    #NTF2 topology should not be used so remove those rows
    ntf2_inx = combined_data[combined_data['topology']=='NTF2'].index
    combined_data = combined_data.drop(ntf2_inx)
    
    col_order = list(combined_data.columns.values)
    col_order.insert(2, col_order.pop(col_order.index('dataset_original')))
    combined_data = combined_data[col_order]
    combined_data['stabilityscore_cnn_calibrated_2classes'] = combined_data['stabilityscore_cnn_calibrated'] > 1

    
    train, test = train_test_split(combined_data, test_size=0.2, random_state=5, stratify=combined_data[['topology', 'dataset_original']])

    # list of feature columns to use and/or normalize:
    ros_fet_path = '/Users/janestrada/Desktop/SorelleSummerResearch/TestHarness/dataframes/rosetta_features.csv'
    spc_fet_path = '/Users/janestrada/Desktop/SorelleSummerResearch/TestHarness/dataframes/entropy_features.csv'

    ros_features = list(pd.read_csv(ros_fet_path).iloc[:,0])
    spc_features = list(pd.read_csv(spc_fet_path).iloc[:,0])
    
    
    feature_cols = ros_features+spc_features

    # TestHarness usage starts here, all code before this was just data input and pre-processing.
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(current_path))
    print()
    th = TestHarness(output_location='/Users/janestrada/Desktop/SorelleSummerResearch/TestHarness')
             
    # ROSETTA + ENTROPY FEATURES
    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
                testing_data=test, data_and_split_description="RS",
                cols_to_predict='stabilityscore_cnn_calibrated_2classes',
                feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                feature_extraction='shap_audit', predict_untested_data=False,interpret_complex_model=True)
    
    # ROSETTA FEATURES                 
    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
                testing_data=test, data_and_split_description="R",
                cols_to_predict='stabilityscore_cnn_calibrated_2classes',
                feature_cols_to_use=ros_features, normalize=True, feature_cols_to_normalize=ros_features,
                feature_extraction='shap_audit', predict_untested_data=False,interpret_complex_model=True)

    # ENTROPY FEATURES
    th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
                testing_data=test, data_and_split_description="S",
                cols_to_predict='stabilityscore_cnn_calibrated_2classes',
                feature_cols_to_use=spc_features, normalize=True, feature_cols_to_normalize=spc_features,
                feature_extraction='shap_audit', predict_untested_data=False,interpret_complex_model=True)
    
if __name__ == '__main__':
    main()


