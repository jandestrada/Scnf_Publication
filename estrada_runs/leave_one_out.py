import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression


# get work directory so that it can run on tacc-resources or local jupyter node
work_dir = None
if os.path.isdir('/home/jupyter/tacc-work/'):
    work_dir = '/home/jupyter/tacc-work/'
elif os.path.isdir('/work/05689/jestrada/'):
    work_dir = '/work/05689/jestrada/'
    
    
VERSIONED_DATA = os.path.join(work_dir,'test-harness-v3/versioned-datasets/data')

print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."
print()


def main():
    # Reading in data from versioned-datasets repo.
    file_path = os.path.join(work_dir,'model_building/dataframes/df_aggregated_spc.csv')
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

    # Grouping Dataframe read in for leave-one-out analysis.
    grouping_df = pd.read_csv(os.path.join(VERSIONED_DATA, 'protein-design/metadata/protein_groupings_by_JE.metadata.csv'), comment='#',
                              low_memory=False)

    # list of feature columns to use and/or normalize:
    ros_fet_path = os.path.join(work_dir,"model_building/rosetta_features.csv")
    spc_fet_path = os.path.join(work_dir,"model_building/entropy_features.csv")
    
    ros_features = list(pd.read_csv(ros_fet_path).iloc[:,0])
    spc_features = list(pd.read_csv(spc_fet_path).iloc[:,0])
    
    
    feature_cols = ros_features+spc_features

    # TestHarness usage starts here, all code before this was just data input and pre-processing.
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(current_path))
    print()
    th = TestHarness(output_location=os.path.join(work_dir,'model_building/estrada_runs/leave_one_out_runs'))
    
    # CLASSIFIERS
#     for i in range(100):
#         # ROSETTA + ENTROPY FEATURES
#         th.run_leave_one_out(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, 
#                              data=combined_data,
#                              data_description="RS", grouping=grouping_df,
#                              grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated_2classes", feature_cols_to_use=feature_cols,
#                              normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)
#         # ROSETTA FEATURES                 
#         th.run_leave_one_out(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, data=combined_data,
#                              data_description="R", grouping=grouping_df,
#                              grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated_2classes", feature_cols_to_use=ros_features,
#                              normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)

#         # ENTROPY FEATURES
#         th.run_leave_one_out(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, data=combined_data,
#                              data_description="S", grouping=grouping_df,
#                              grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated_2classes", feature_cols_to_use=spc_features,
#                              normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)
        
        
     # REGRESSORS
    for i in range(100):
        # ROSETTA + ENTROPY FEATURES
        th.run_leave_one_out(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, 
                             data=combined_data.dropna(axis=1),
                             data_description="RS", grouping=grouping_df,
                             grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated",
                             feature_cols_to_use=feature_cols,
                             normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)
        # ROSETTA FEATURES                 
        th.run_leave_one_out(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, 
                             data=combined_data,
                             data_description="R", grouping=grouping_df,
                             grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated", 
                             feature_cols_to_use=ros_features,
                             normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)

        # ENTROPY FEATURES
        th.run_leave_one_out(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, 
                             data=combined_data,
                             data_description="S", grouping=grouping_df,
                             grouping_description="grouping_JE", cols_to_predict="stabilityscore_cnn_calibrated", 
                             feature_cols_to_use=spc_features,
                             normalize=True, feature_cols_to_normalize=feature_cols, feature_extraction=False)

if __name__ == '__main__':
    main()
