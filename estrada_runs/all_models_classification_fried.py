import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
#from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
    
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.keras_classification import keras_classification_4, keras_classification_ros_spc, keras_classification_spc,keras_classification_ros
from harness.th_model_instances.estrada_models.decision_tree_classification import decision_tree_classification
from harness.th_model_instances.estrada_models.gbc_classification import gbc_classification
from harness.th_model_instances.estrada_models.gmm_classification import gmm_classification
from harness.th_model_instances.estrada_models.knn_classification import knn_classifier
from harness.th_model_instances.estrada_models.naive_bayes_classification import naive_bayes_classification
from harness.th_model_instances.estrada_models.svm_classification import svm_classification



def main():
    # Reading in data from versioned-datasets repo.
    file_path = '/home/h205c/jestrada/Scnf_Publication/data/df_aggregated_spc.csv'
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
    ros_fet_path = '/home/h205c/jestrada/Scnf_Publication/data/rosetta_features.csv'
    spc_fet_path = '/home/h205c/jestrada/Scnf_Publication/data/entropy_features.csv'

    ros_features = list(pd.read_csv(ros_fet_path).iloc[:,0])
    spc_features = list(pd.read_csv(spc_fet_path).iloc[:,0])
    
    
    feature_cols = ros_features+spc_features

    # TestHarness usage starts here, all code before this was just data input and pre-processing.
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(
        '/home/h205c/jestrada/Scnf_Publication/data/TestHarness'))
    print()
    th = TestHarness(output_location='/home/h205c/jestrada/Scnf_Publication/data/TestHarness')

    #make a list of all models to be used
    models_to_be_used = [(random_forest_classification,'discreet'),(keras_classification_4,'discreet'), (decision_tree_classification,'discreet'),
                        (gbc_classification,'discreet'), (gmm_classification,'discreet'), (knn_classifier,'discreet'), (naive_bayes_classification,'discreet'),
                        (svm_classification,'discreet')]



    for i,j in models_to_be_used:
        model = i
        target_col = None
        model_spc = None
        model_ros_spc = None
        model_ros = None
        print(f"USING {model} MODEL.")
        if i==keras_classification_4: 
            model_spc = keras_classification_spc
            model_ros_spc = keras_classification_ros_spc
            model_ros = keras_classification_ros
        else:
            model_spc = model
            model_ros_spc = model
            model_ros = model

        if j=='continuous':
            target_col = 'stabilityscore_cnn_calibrated'
        if j=='discreet':
            target_col = 'stabilityscore_cnn_calibrated_2classes'

        # ROSETTA + ENTROPY FEATURES
        th.run_custom(function_that_returns_TH_model=model_ros_spc, dict_of_function_parameters={}, training_data=train,
                    testing_data=test, data_and_split_description="RS",
                    cols_to_predict=target_col,
                    feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                    feature_extraction=False, predict_untested_data=False,interpret_complex_model=False)
        
        # ROSETTA FEATURES                 
        th.run_custom(function_that_returns_TH_model=model_ros, dict_of_function_parameters={}, training_data=train,
                    testing_data=test, data_and_split_description="R",
                    cols_to_predict=target_col,
                    feature_cols_to_use=ros_features, normalize=True, feature_cols_to_normalize=ros_features,
                    feature_extraction=False, predict_untested_data=False,interpret_complex_model=False)

        # ENTROPY FEATURES
        th.run_custom(function_that_returns_TH_model=model_spc, dict_of_function_parameters={}, training_data=train,
                    testing_data=test, data_and_split_description="S",
                    cols_to_predict=target_col,
                    feature_cols_to_use=spc_features, normalize=True, feature_cols_to_normalize=spc_features,
                    feature_extraction=False, predict_untested_data=False,interpret_complex_model=False)
    
if __name__ == '__main__':
    main()


