import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression

# At some point in your script you will need to define your data. For most cases the data will come from the `versioned_datasets` repo,
# which is why in this example script I am pointing to the data folder in the `versioned-datasets` repo:
# Ideally you would clone the `versioned-datasets` repo in the same location where you cloned the `protein-design` repo,
# but it shouldn't matter as long as you put the correct path here.

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

    # At some point in your script you will need to define your data. For most cases the data will come from the `versioned_datasets` repo,
    # which is why in this example script I am pointing to the data folder in the `versioned-datasets` repo:
    # Ideally you would clone the `versioned-datasets` repo in the same location where you cloned the `protein-design` repo,
    # but it shouldn't matter as long as you put the correct path here.
    print("Path to data folder in the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATA))
    assert os.path.isdir(VERSIONED_DATA), "The path you gave for VERSIONED_DATA does not exist."
    print()


    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    combined_data = pd.read_csv(os.path.join(work_dir,'model_building/dataframes/df_aggregated_spc.csv'))
    combined_data['dataset_original'] = combined_data['dataset']
    
    #NTF2 topology should not be used so remove those rows
    ntf2_inx = combined_data[combined_data['topology']=='NTF2'].index
    combined_data = combined_data.drop(ntf2_inx)

    col_order = list(combined_data.columns.values)
    col_order.insert(2, col_order.pop(col_order.index('dataset_original')))
    combined_data = combined_data[col_order]
    combined_data['stabilityscore_cnn_calibrated_2classes'] = combined_data['stabilityscore_cnn_calibrated'] > 1

    # list of feature columns to use and/or normalize:
    ros_features = list(pd.read_csv(os.path.join(work_dir,"model_building/rosetta_features.csv")).iloc[:,0])
    spc_features = list(pd.read_csv(os.path.join(work_dir,"model_building/entropy_features.csv")).iloc[:,0])
    feature_cols = ros_features+spc_features

    #get list of topologies
    #topology_list = list(combined_data['topology'].value_counts().index)
    topology_list = ['thio']
    #for n in range(100):
        # Run the following on each topology

    for i in range(len(topology_list)):
        print("Currently in %s topology"%topology_list[i])

        combined_data_specified = combined_data[combined_data['topology']==topology_list[i]]
        current_path = os.getcwd()
        print("initializing TestHarness object with output_location equal to {}".format(current_path))
        print()
        th = TestHarness(output_location=os.path.join(work_dir,'model_building/topology_specific_runs'))
        train, test = train_test_split(combined_data_specified, test_size=0.2, random_state=5, stratify=combined_data_specified['dataset_original'])


#         # ROSETTA + ENTROPY FEATURES
#         th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
#                       testing_data=test, data_and_split_description="RS %s"%topology_list[i],
#                       cols_to_predict='stabilityscore_cnn_calibrated_2classes',
#                       feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
#                       feature_extraction=False, predict_untested_data=False)
#         # ROSETTA FEATURES                 
#         th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
#                       testing_data=test, data_and_split_description="R %s"%topology_list[i],
#                       cols_to_predict='stabilityscore_cnn_calibrated_2classes',
#                       feature_cols_to_use=ros_features, normalize=True, feature_cols_to_normalize=ros_features,
#                       feature_extraction=False, predict_untested_data=False)

#         # ENTROPY FEATURES
#         th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={}, training_data=train,
#                       testing_data=test, data_and_split_description="S %s"%topology_list[i],
#                       cols_to_predict='stabilityscore_cnn_calibrated_2classes',
#                       feature_cols_to_use=spc_features, normalize=True, feature_cols_to_normalize=spc_features,
#                       feature_extraction=False, predict_untested_data=False)

       # ROSETTA + ENTROPY FEATURES
    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, training_data=train,
                  testing_data=test, data_and_split_description="RS %s"%topology_list[i],
                  cols_to_predict='stabilityscore_cnn_calibrated',
                  feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                  feature_extraction=False, predict_untested_data=test)
    # ROSETTA FEATURES                 
    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, training_data=train,
                  testing_data=test, data_and_split_description="R %s"%topology_list[i],
                  cols_to_predict='stabilityscore_cnn_calibrated',
                  feature_cols_to_use=ros_features, normalize=True, feature_cols_to_normalize=ros_features,
                  feature_extraction=False, predict_untested_data=test)

    # ENTROPY FEATURES
    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, training_data=train,
                  testing_data=test, data_and_split_description="S %s"%topology_list[i],
                  cols_to_predict='stabilityscore_cnn_calibrated',
                  feature_cols_to_use=spc_features, normalize=True, feature_cols_to_normalize=spc_features,
                  feature_extraction=False, predict_untested_data=test)


if __name__ == '__main__':
    main()
