import json
from datetime import datetime
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.model_selection import train_test_split

from harness.test_harness_class import TestHarness
#from version import VERSION
from scripts_for_automation.perovskite_models_config import MODELS_TO_RUN
from harness.utils.object_type_modifiers_and_checkers import make_list_if_not_list

import warnings
#import git
import re

warnings.filterwarnings("ignore")

PREDICTED_OUT = "predicted_out"
SCORE = "score"
RANKING = "ranking"
# how many predictions from the test harness to send to submissions server
NUM_PREDICTIONS = 100

# todo: oops, committed this.  Need to revoke, but leaving for testing
AUTH_TOKEN = '4a8751b83c9744234367b52c58f4c46a53f5d0e0225da3f9c32ed238b7f82a69'

ESCALATION_SERVER_DEV = 'http://escalation-dev.sd2e.org'
ESCALATION_SERVER = "http://escalation.sd2e.org"


def get_git_commit_id():
    # using Path(__file__).parents[1] to get the path of the directory immediately above this file's directory
    repo = git.Repo(Path(__file__).parents[1])
    git_sha = repo.head.object.hexsha
    return git_sha[0:7]


# compute md5 hash using small chunks
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_truncated_float(float_number, n=5):
    # By decision, we want to output reagent values as floats truncated (not rounded!) to 5 decimal places
    s = '{}'.format(float_number)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def get_prediction_csvs(run_ids, predictions_csv_path=None):
    prediction_csv_paths = []
    if predictions_csv_path is None:
        runs_path = os.path.join('/tacc-work/model_building/estrada_runs/perovskite_runs/test_harness_results', 'runs')
        previous_runs = []
        for this_run_folder in os.listdir(runs_path):
            if this_run_folder.rsplit("_")[1] in run_ids:
                print('{} was kicked off by this TestHarness instance. Its results will be submitted.'.format(this_run_folder))
                prediction_csv_path = os.path.join(runs_path, this_run_folder, 'predicted_data.csv')
                if os.path.exists(prediction_csv_path):
                    print("file found: ", prediction_csv_path)
                    prediction_csv_paths.append(prediction_csv_path)
            else:
                previous_runs.append(this_run_folder)
        print('\nThe results for the following runs will not be submitted, '
              'because they are older runs that were not initiated by this TestHarness instance:'
              '\n{}\n'.format(previous_runs))
    else:
        prediction_csv_paths.append(predictions_csv_path)
    return prediction_csv_paths


def select_which_predictions_to_submit(predictions_df, all_or_subset='subset'):
    # use binarized predictions, change 1s to 4s, and 0s to 1s
    all_preds = predictions_df.copy()
    all_preds.loc[all_preds[PREDICTED_OUT] == 1, PREDICTED_OUT] = 4
    all_preds.loc[all_preds[PREDICTED_OUT] == 0, PREDICTED_OUT] = 1
    all_preds.sort_values(by=RANKING, ascending=True, inplace=True)
    all_preds.reset_index(inplace=True, drop=True)

    # drop ranking column to match file format of the submission server
    all_preds.drop(axis=1, columns=RANKING, inplace=True)

    if all_or_subset == 'subset':
        return all_preds.head(NUM_PREDICTIONS)
    elif all_or_subset == 'all':
        return all_preds
    else:
        raise ValueError("all_or_subset must equal 'subset' or 'all'")


def build_submissions_csvs_from_test_harness_output(prediction_csv_paths, crank_number, commit_id):
    submissions_paths = []
    for prediction_path in prediction_csv_paths:
        # todo: we need to know about what model this was for the notes field and such
        columns = {"dataset": "dataset",
                   "name": "name",
                   "_rxn_M_inorganic": "_rxn_M_inorganic",
                   "_rxn_M_organic": "_rxn_M_organic",
                   "_rxn_M_acid": "_rxn_M_acid",
                   "binarized_crystalscore_predictions": PREDICTED_OUT,
                   "binarized_crystalscore_prob_predictions": SCORE,
                   "binarized_crystalscore_rankings": RANKING}
        df = pd.read_csv(prediction_path, comment='#')
        df = df.filter(columns.keys())
        df = df.rename(columns=columns)
        df['dataset'] = crank_number
        selected_predictions = select_which_predictions_to_submit(predictions_df=df, all_or_subset='subset')

        # fix formatting
        # truncate floats to 5 digits
        # for column in ['_rxn_M_inorganic', '_rxn_M_organic']:
        #     selected_predictions[column] = selected_predictions[column].apply(format_truncated_float)
        # 0-pad crank number if padding has been removed
        # selected_predictions['dataset'] = selected_predictions['dataset'].apply(lambda x: '{0:0>4}'.format(x))
        username = 'testharness'
        submission_template_filename = '_'.join([crank_number,
                                                 'train',
                                                 commit_id,
                                                 username]) + '.csv'
        submissions_file_path = os.path.join(os.path.dirname(prediction_path), submission_template_filename)

        selected_predictions.to_csv(submissions_file_path, index=False)
        submissions_paths.append(submissions_file_path)
    return submissions_paths


def submit_csv_to_escalation_server(submissions_file_path, crank_number, commit_id, escalation_server=ESCALATION_SERVER_DEV):
    test_harness_results_path = submissions_file_path.rsplit("/runs/")[0]
    this_run_results_path = submissions_file_path.rsplit("/", 1)[0]

    leaderboard = pd.read_html(os.path.join(test_harness_results_path, 'custom_classification_leaderboard.html'))[0]
    leaderboard_entry_for_this_run = leaderboard.loc[leaderboard["Run ID"] == this_run_results_path.rsplit("/run_")[1]]

    model_name = leaderboard_entry_for_this_run["Model Name"].values[0]
    model_author = leaderboard_entry_for_this_run["Model Author"].values[0]
    model_description = leaderboard_entry_for_this_run["Model Description"].values[0]

    response = requests.post(escalation_server + "/submission",
                             headers={'User-Agent': 'escalation'},
                             data={'crank': crank_number,
                                   'username': "test_harness_{}".format(VERSION),
                                   'expname': model_name,
                                   'githash': commit_id,
                                   # todo: add check to make sure that notes doesn't contain any commas
                                   'notes': "Model Author: {}; "
                                            "Model Description: {}; "
                                            "Test Harness Hash: {}; "
                                            "Submitted at {}".format(model_author, model_description, get_git_commit_id(),
                                                                     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))},
                             files={'csvfile': open(submissions_file_path, 'rb')},
                             # timeout=60
                             )
    print("Submitted file to submissions server")
    return response, response.text


def build_leaderboard_rows_dict(submissions_file_path, crank_number):
    """
    :param submissions_file_path: a file path string
    :return: As dict keyed by run_id, valued with the row from the leaderboard
    """
    test_harness_results_path = submissions_file_path.rsplit("/runs/")[0]
    leaderboard = pd.read_html(os.path.join(test_harness_results_path, 'custom_classification_leaderboard.html'))[0]
    leaderboard["Dataset"] = crank_number
    leaderboard.columns = [x.lower().replace(' ', '_') for x in leaderboard.columns]
    leaderboard_rows_dict = leaderboard.set_index('run_id', drop=False).to_dict(orient='index')
    return leaderboard_rows_dict


def submit_leaderboard_to_escalation_server(leaderboard_rows_dict, submission_path, commit_id, escalation_server=ESCALATION_SERVER_DEV):
    # gets run id from path of form 'test_harness_results/runs/run_aXRQm2Ox6RY7m/0021_train_323354d_testharness.csv'
    # This is kind of brittle.
    run_id = submission_path.split('/')[2].split('_')[1]
    row = leaderboard_rows_dict[run_id]
    row['githash'] = commit_id
    response = requests.post(escalation_server + "/leaderboard",
                             headers={'User-Agent': 'escalation'},
                             data=row,
                             timeout=60
                             )
    if response.status_code != 200:
        print('Error submitting leaderboard row:', row, response.text)


def get_crank_number_from_filename(training_data_filename):
    # Gets the challenge problem iteration number from the training data
    crank_number = os.path.basename(training_data_filename).split('.')[0]
    # should be of format ####
    assert len(crank_number) == 4
    return crank_number


def run_configured_test_harness_models_on_perovskites(train_set, state_set):
    all_cols = train_set.columns.tolist()
    # don't worry about _calc_ columns for now, but it's in the code so they get included once the data is available
    feature_cols = [c for c in all_cols if ("_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (train_set.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # print(set(state_set.columns.tolist()).difference(set(feature_cols)))
    # print(set(feature_cols).difference(set(state_set.columns.tolist())))
    # remove _rxn_temperatureC_actual_bulk column from feature_cols because it doesn't exist in state_set
    feature_cols.remove("_rxn_temperatureC_actual_bulk")

    # create binarized crystal scores because Ian said to start with binary task
    # also multiclass support needs to be added to Test Harness
    conditions = [
        (train_set['_out_crystalscore'] == 1),
        (train_set['_out_crystalscore'] == 2),
        (train_set['_out_crystalscore'] == 3),
        (train_set['_out_crystalscore'] == 4),
    ]
    binarized_labels = [0, 0, 0, 1]
    train_set['binarized_crystalscore'] = np.select(conditions, binarized_labels)
    col_order = list(train_set.columns.values)
    col_order.insert(3, col_order.pop(col_order.index('binarized_crystalscore')))
    train_set = train_set[col_order]

    col_to_predict = 'binarized_crystalscore'

    train, test = train_test_split(train_set, test_size=0.2, random_state=5, stratify=train_set[['dataset']])

    # Test Harness use starts here:
    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}\n".format(current_path))
    th = TestHarness(output_location=os.path.join(current_path,'perovskite_runs'), output_csvs_of_leaderboards=True)

    for model in MODELS_TO_RUN:
        th.run_custom(function_that_returns_TH_model=model, dict_of_function_parameters={},
                      training_data=train,
                      testing_data=test, data_and_split_description="test run on perovskite data",
                      cols_to_predict=col_to_predict,
                      feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                      feature_extraction='bba_audit', predict_untested_data=state_set,
                      index_cols=["dataset", "name", "_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]
                      )

    return th.list_of_this_instance_run_ids


def get_manifest_from_gitlab_api(commit_id, auth_token):
    headers = {"Authorization": "Bearer {}".format(auth_token)}
    # this is the API call for the versioned data repository.  It gets the raw data file.
    # 202 is the project id, derived from a previous call to the projects endpoint
    # we have hard code the file we are fetching (manifest/perovskite.manifest.yml), and vary the commit id to fetch
    gitlab_manifest_url = \
        'https://gitlab.sd2e.org/api/v4/projects/202/repository/files/manifest%2fperovskite.manifest.yml/raw?ref={}'.format(commit_id)
    response = requests.get(gitlab_manifest_url, headers=headers)
    if response.status_code == 404:
        raise KeyError("File perovskite manifest not found from Gitlab API for commit {}".format(commit_id))
    elif response.status_code == 403:
        raise RuntimeError("Unable to authenticate user with gitlab")
    perovskite_manifest = yaml.load(response.text)
    return perovskite_manifest


def get_git_hash_at_versioned_data_master_tip(auth_token):
    headers = {"Authorization": "Bearer {}".format(auth_token)}
    # this is the API call for the versioned data repository.  It gets the raw data file.
    # 202 is the project id, derived from a previous call to the projects endpoint
    # we have hard code the file we are fetching (manifest/perovskite.manifest.yml), and vary the commit id to fetch
    gitlab_manifest_url = 'https://gitlab.sd2e.org/api/v4//projects/202/repository/commits/master'
    response = requests.get(gitlab_manifest_url, headers=headers)
    if response.status_code == 404:
        raise KeyError("Unable to find metadata on master branch of versioned data repo via gitlab API")
    elif response.status_code == 403:
        raise RuntimeError("Unable to authenticate user with gitlab")
    gitlab_master_branch_metadata = json.loads(response.text)
    tip_commit_id = gitlab_master_branch_metadata["id"][:7]
    return tip_commit_id


def is_list_of_crank_strings(obj):
    if obj and isinstance(obj, list):
        # re.compile.match ensures that the string passed in follows the format of four integers in a string
        return all(re.compile("^[0-9]{4}$").match(elem) for elem in obj)
    else:
        return False


def get_all_training_and_stateset_filenames(manifest):
    """
    Takes a manifest and finds all the perovskitedata and stateset files listed inside
    :param manifest: dict of perovskite manifest file
    :return: dictionary of perovskitedata and stateset file paths
    """
    files_of_interest = {'perovskitedata': [], 'stateset': []}
    for file_name in manifest['files']:
        for file_type, existing_filenames in files_of_interest.items():
            if file_name.endswith('{}.csv'.format(file_type)):
                existing_filenames.append(file_name)

    for file_type, existing_filenames in files_of_interest.items():
        assert len(existing_filenames) > 0, "No file found in manifest for type {}".format(file_type)

    return files_of_interest


def get_latest_training_and_stateset_filenames(manifest):
    """
    Takes a manifest and finds the most recent (highest crank number) perovskite data file and stateset file.
    :param manifest: dict of perovskite manifest file
    :return: paths to perovskite_data_file and stateset_file
    """
    files_of_interest = get_all_training_and_stateset_filenames(manifest)
    # get the files of interest with the highest crank number, assert crank numbers are equal
    perovskite_data_file = sorted(files_of_interest['perovskitedata'], reverse=True)[0]
    stateset_file = sorted(files_of_interest['stateset'], reverse=True)[0]
    assert get_crank_number_from_filename(stateset_file) == get_crank_number_from_filename(perovskite_data_file)
    return perovskite_data_file, stateset_file


def get_crank_specific_training_and_stateset_filenames(manifest, specific_crank_number):
    """
    Takes a manifest and finds the perovskite data file and stateset file associated with a specific crank number.
    :param manifest: dict of perovskite manifest file
    :param specific_crank_number: string that looks like "0022", representing the crank number you want filenames for
    :return: paths to perovskitedata_file and stateset_file for the specific_crank_number
    """
    files_of_interest = get_all_training_and_stateset_filenames(manifest)

    perovskitedata_files = [x for x in files_of_interest['perovskitedata'] if get_crank_number_from_filename(x) == specific_crank_number]
    stateset_files = [x for x in files_of_interest['stateset'] if get_crank_number_from_filename(x) == specific_crank_number]

    if len(perovskitedata_files) == 0:
        raise ValueError("The specific_crank_number ({}) that was passed in does not exist in any listed "
                         "perovskitedata file in the manifest. Make sure your value for specific_crank_number "
                         "is of the format '0019' and exists in the manifest.".format(specific_crank_number))
    elif len(perovskitedata_files) > 1:
        raise ValueError("It appears that the manifest has multiple perovskitedata files with the same specific_crank_number."
                         "There is likely an issue with the manifest.")
    else:
        perovskitedata_file = perovskitedata_files[0]

    if len(stateset_files) == 0:
        raise ValueError("The specific_crank_number that was passed in does not exist in any listed stateset file in the manifest. "
                         "Make sure your value for specific_crank_number is of the format '0019' and exists in the manifest.")
    elif len(stateset_files) > 1:
        raise ValueError("It appears that the manifest has multiple stateset files with the same specific_crank_number."
                         "There is likely an issue with the manifest.")
    else:
        stateset_file = stateset_files[0]

    assert get_crank_number_from_filename(stateset_file) == get_crank_number_from_filename(perovskitedata_file)
    return perovskitedata_file, stateset_file


def run_cranks(versioned_data_path, cranks="latest"):
    manifest_file = os.path.join(versioned_data_path, "manifest/perovskite.manifest.yml")
    with open(manifest_file) as f:
        manifest_dict = yaml.load(f)

    perovskite_data_folder_path = os.path.join(versioned_data_path, "data/perovskite")

    if cranks == "latest":
        training_data_filename, state_set_filename = get_latest_training_and_stateset_filenames(manifest_dict)
        training_state_tuples = list(zip([training_data_filename], [state_set_filename]))
    elif cranks == "all":
        all_files_dict = get_all_training_and_stateset_filenames(manifest_dict)
        perovskitedata_files = sorted(all_files_dict['perovskitedata'], reverse=False)
        stateset_files = sorted(all_files_dict['stateset'], reverse=False)
        training_state_tuples = list(zip(perovskitedata_files, stateset_files))
    else:
        cranks = make_list_if_not_list(cranks)
        assert is_list_of_crank_strings(cranks), \
            "cranks must equal 'latest', 'all', or a string (or list of strings) of format '0021' that represent(s) a specific crank."
        print("Will run the following {} cranks: {}\n".format(len(cranks), cranks))

        training_state_tuples = []
        for c in cranks:
            training_data_filename, state_set_filename = get_crank_specific_training_and_stateset_filenames(manifest_dict, c)
            training_state_tuples.append((training_data_filename, state_set_filename))

    print("\ntraining_state_tuples being passed to crank_runner:\n{}\n".format(training_state_tuples.copy()))
    for training_data_filename, state_set_filename in training_state_tuples:
        assert get_crank_number_from_filename(training_data_filename) == get_crank_number_from_filename(state_set_filename)
        training_data_path = os.path.join(perovskite_data_folder_path, training_data_filename)
        state_set_path = os.path.join(perovskite_data_folder_path, state_set_filename)
        crank_runner(training_data_path, state_set_path)


def crank_runner(training_data_path, state_set_path):
    crank_number = get_crank_number_from_filename(training_data_path)
    print("\nRunning Crank {}".format(crank_number))
    print("Crank {} Training Data Path: {}".format(crank_number, training_data_path))
    print("Crank {} State Set Path: {}".format(crank_number, state_set_path))
    print()

    training_data = pd.read_csv(training_data_path, comment='#', low_memory=False)
    state_set = pd.read_csv(state_set_path, comment='#', low_memory=False)

    list_of_run_ids = run_configured_test_harness_models_on_perovskites(training_data, state_set)

    # this uses current master commit on the origin
    commit_id = get_git_hash_at_versioned_data_master_tip(AUTH_TOKEN)
    prediction_csv_paths = get_prediction_csvs(run_ids=list_of_run_ids)
    submissions_paths = build_submissions_csvs_from_test_harness_output(prediction_csv_paths,
                                                                        crank_number,
                                                                        commit_id)
    if submissions_paths:
        # If there were any submissions, include the leaderboard
        # Only one leaderboard file is made, so we can submit just by pointing one path
        submissions_path = submissions_paths[0]
        leaderboard_rows_dict = build_leaderboard_rows_dict(submissions_path, crank_number)
    for submission_path in submissions_paths:
        print("Submitting {} to escalation server".format(submission_path))
        response, response_text = submit_csv_to_escalation_server(submission_path, crank_number, commit_id)
        print("Submission result: {}".format(response_text))
        submit_leaderboard_to_escalation_server(leaderboard_rows_dict, submission_path, commit_id)


if __name__ == '__main__':
    """
    NB: This script is for local testing, and is NOT what is run by the app.
    The test harness app runs in perovskite_test_harness.py
    """
    VERSIONED_DATASETS = '/home/jupyter/tacc-work/test-harness-v3/versioned-datasets'
    print("Path to the locally cloned versioned-datasets repo was set to: {}".format(VERSIONED_DATASETS))
    print()
    assert os.path.isdir(VERSIONED_DATASETS), "The path you gave for VERSIONED_DATA does not exist."

    # set cranks equal to "latest", "all", or a string of format '0021' representing a specific crank number
    run_cranks(VERSIONED_DATASETS, cranks='latest')

# todo: round instead of truncate float
