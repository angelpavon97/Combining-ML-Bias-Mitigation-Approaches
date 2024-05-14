import pandas as pd
import numpy as np
import json
from aif360.datasets import StandardDataset, BinaryLabelDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

from statistical_measures import get_mannwhitneyu, apply_categorical_tests
from machine_learning import search_best_attributes


def convert_X_to_aif360(X, y, sensitive_attribute, favorable_class=1, privileged_groups = [0]):
    df = pd.concat([X, y], axis=1)

    aif360_df = StandardDataset(df, 
                          label_name='Class', 
                          favorable_classes=[favorable_class], 
                          protected_attribute_names=[sensitive_attribute], 
                          privileged_classes=[privileged_groups])

    # aif360_df = BinaryLabelDataset(df=df, 
    #                       label_names=[target], 
    #                       protected_attribute_names=[sensitive_attribute],
    #                       favorable_label=favorable_class,
    #                       unfavorable_label=unfavorable_class)

    return aif360_df

def convert_aif360_to_X(aif360_df, target = 'Class'):

    df = aif360_df.convert_to_dataframe()[0]

    y = df[target]
    X = df.drop(columns=[target])

    # Convert index to int64
    X.index = X.index.astype('int64')
    y.index = y.index.astype('int64')
    
    return X,y

def get_attributes_covariance(df, sensitive_attribute):
    _, chi_covariance_attributes = apply_categorical_tests(df, class_name = sensitive_attribute, alpha = -1)
    _, mu_covariance_attributes = get_mannwhitneyu(df, class_name = sensitive_attribute, alpha = -1)

    attributes_covariance = {**chi_covariance_attributes, **mu_covariance_attributes}

    return attributes_covariance

def get_covariance_proxies(df, sensitive_attribute, alpha = 0.01):

    dependent_attributes, independent_attributes = apply_categorical_tests(df, class_name = sensitive_attribute, alpha = alpha)
    diff_dis_attributes, same_dis_attributes = get_mannwhitneyu(df, class_name = sensitive_attribute, alpha = alpha)

    dependent_attributes = {**dependent_attributes, **diff_dis_attributes}

    return dependent_attributes
    
def covariance_mitigation(X, sensitive_attribute = None, df_clean = None, attributes_covariance = None, alpha = 0.01, removing_errors='ignore'):

    if attributes_covariance is None:
        if df_clean is None or sensitive_attribute is None:
            raise ValueError('If attributes_covariance are not provided, sensitive_attribute and df_clean must be provided to extract the proxies.')
        else:
            covariance_proxies = get_covariance_proxies(df_clean, sensitive_attribute, alpha)
    else:
        covariance_proxies = [attribute for attribute, value in attributes_covariance.items() if value < alpha]

    X = X.drop(columns=covariance_proxies, errors=removing_errors)

    return X

def get_alpha_covariance(df, attributes_covariance, sensitive_attribute, class_name = 'Class', start_alpha = 0.01, stop_alpha = 0.11, step_alpha = 0.01, s_prediction_error = 0.01, verbose = False):
    
    majority_class_percentage = max(df[sensitive_attribute].value_counts() / len(df))

    alphas = np.arange(start_alpha, stop_alpha, step_alpha).tolist()
    df = df.drop(columns = [class_name])
    
    for a in alphas:
        df2 = df.copy()
        covariance_proxies = [attribute for attribute, value in attributes_covariance.items() if value < a]

        df2 = df2.drop(columns=covariance_proxies, errors='ignore')
        df2[sensitive_attribute] = df[sensitive_attribute]

        best_clf, best_attributes, best_acc = search_best_attributes(df2, class_name = sensitive_attribute, clf=XGBClassifier())

        if best_acc <= majority_class_percentage + s_prediction_error:
            if verbose:
                print(f'alpha for Covariance method: {a}\nBest accuracy: {best_acc} (Majority class: {majority_class_percentage})')
            return a

    return a

def set_new_estimator(params, estimator):
    if 'estimator' in params.keys():
        params['estimator'] = estimator
        params['prefit'] = True
    
    return params

def test_model(clf, X_test, sensitive_attribute, mitigation_type = None, y_test = None, favorable_class=1, privileged_groups = [0], target = 'Class'):
    if mitigation_type != 'in':
        return clf.predict(X_test)
    else:
        aif360_df_test = convert_X_to_aif360(X_test, y_test, sensitive_attribute, 
                                             favorable_class=favorable_class, privileged_groups=privileged_groups)
        aif360_result_df = clf.predict(aif360_df_test)
        X_res, y_res = convert_aif360_to_X(aif360_result_df, target=target)
        return y_res
    
def performance_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred)
    f1_weighted_score = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    auc_score = roc_auc_score(y_true, y_pred)

    results = {
        'Accuracy': accuracy,
        'F1 Score': f1_score_value,
        'F1 Weighted': f1_weighted_score,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc_score
    }

    return results

def store_results_json(results, path):
    with open(path, 'w') as json_file:
        json.dump(results, json_file)

    # Load the JSON file back into a dictionary to check that it was saved correctly
    with open(path, 'r') as json_file:
        loaded_dict = json.load(json_file)
    
    return loaded_dict
    