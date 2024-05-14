import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fairness_utils import get_alpha_covariance, get_attributes_covariance, store_results_json
from preprocessing import get_data, clean_data, process_data, clean_df_census, process_data_census
from fairness_comparative import get_bias_mitigation_result
from hybrid_fairness import get_hybrid_bias_mitigation_result, get_proxies_mitigation_results
import json
import os
from folktables import ACSDataSource, BasicProblem, adult_filter

def get_german_data():
    df = get_data('./data/german.data')
    df = clean_data(df)

    df_preprocessed = process_data(df, encoder = 'Label')
    df_preprocessed = df_preprocessed.astype(int)

    return df, df_preprocessed

def get_folktablesIncome_data(state = 'NY', year = 2018):
    STATES = [state]
    YEAR = str(year)

    data_source = ACSDataSource(survey_year=YEAR, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=STATES, download=True)

    feature_codes = list(acs_data.columns)

    ACSIncome = BasicProblem(
        features=[feature_codes],
        target='PINCP',
        target_transform=lambda x: x > 50000,    
        group='SEX',
        preprocess=adult_filter,
        postprocess=lambda x: x,
    )

    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.concat([pd.DataFrame(features), pd.DataFrame(label)], axis=1)
    df.columns = feature_codes + ['TARGET']

    definition_df = data_source.get_definitions(download=True)
    df = clean_df_census(df, definition_df)
    
    df_preprocessed = process_data_census(df)
    df_preprocessed = df_preprocessed.astype(int)

    df = df.rename(columns={'SEX':'gender'})

    return df, df_preprocessed


def is_hybrid_exception(mitigation1, mitigation2):
    return False
    # # if 'LFR' in  mitigation1['method'] or 'LFR' in mitigation2['method']:
    # #     return True
    # if 'ThresholdOptimizer' in mitigation1['method'] and mitigation2['type'] == 'in':
    #     # ThresholdOptimizer needs a sklearn classifier
    #     return True
    # elif mitigation1['type'] == 'in' and 'ThresholdOptimizer' in mitigation2['method']:
    #     # ThresholdOptimizer needs a sklearn classifier
    #     return True
    # else:
    #     return False

def get_presaved_methods(sensitive_attribute, base_estimator = RandomForestClassifier(), privileged_groups = [0], unprivileged_groups = [1], attributes_covariance = None, alpha_covariance = None):
    pre_mitigation_methods = [{'method':'Unawareness', 'params':{}},
                          {'method':'DisparateImpactRemover', 'params':{'sensitive_attribute':sensitive_attribute, 'repair_level':0.5}},
                          {'method':'Reweighing', 'params':{'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                          {'method':'LFR', 'params':{'k':5, 'Ax':1.0, 'Ay':1.0, 'Az':1.0,'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}, # New
                          {'method':'LFRDefault', 'params':{'k':5, 'Ax':0.01, 'Ay':1.0, 'Az':50.0,'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}, # New
                          {'method':'Covariance', 'params':{'attributes_covariance':attributes_covariance, 'alpha':alpha_covariance}},
                          {'method':'Covariance01', 'params':{'attributes_covariance':attributes_covariance, 'alpha':0.01}},
                          {'method':'Covariance05', 'params':{'attributes_covariance':attributes_covariance, 'alpha':0.05}},
                          {'method':'Covariance09', 'params':{'attributes_covariance':attributes_covariance, 'alpha':0.09}},
                          {'method':'CorrelationRemover', 'params':{'alpha':0.5, 'sensitive_feature_ids':[sensitive_attribute]}}]
                        #   {'method':'OptimPreproc', 'params':{}}]
    in_mitigation_methods = [{'method':'GerryFairClassifier_FP', 'params':{'fairness_def': 'FP'}},
                         {'method':'GerryFairClassifier_FN', 'params':{'fairness_def': 'FN'}},
                        #  {'method':'GerryFairClassifier_SP', 'params':{'fairness_def': 'SP'}}, SP not supported
                         # {'method':'PrejudiceRemover', 'params':{'sensitive_attr':sensitive_attribute, 'class_attr':target}},
                         {'method':'ExponentiatedGradientReduction_DP', 'params':{'estimator':base_estimator, 'constraints':'DemographicParity'}},
                         {'method':'ExponentiatedGradientReduction_EqOdds', 'params':{'estimator':base_estimator, 'constraints':'EqualizedOdds'}},
                         {'method':'ExponentiatedGradientReduction_TPR', 'params':{'estimator':base_estimator, 'constraints':'TruePositiveRateParity'}},
                         {'method':'ExponentiatedGradientReduction_FPR', 'params':{'estimator':base_estimator, 'constraints':'FalsePositiveRateParity'}},
                         # {'method':'GridSearchReduction', 'params':{'estimator':base_estimator, 'constraints':'DemographicParity'}},
                         {'method':'MetaFairClassifier_DI', 'params':{'sensitive_attr':sensitive_attribute, 'type': 'sr'}},
                         {'method':'MetaFairClassifier_FDR', 'params':{'sensitive_attr':sensitive_attribute, 'type': 'fdr'}},
                         {'method':'AdversialDebiasing', 'params':{'scope_name':'classifier','sess':tf.compat.v1.Session(),'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}]
    post_mitigation_methods = [{'method':'RejectOptionClassification_SP', 'params':{'metric_name':'Statistical parity difference','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                           {'method':'RejectOptionClassification_AvgOdds', 'params':{'metric_name':'Average odds difference','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                           {'method':'RejectOptionClassification_EqOpp', 'params':{'metric_name':'Equal opportunity difference','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                          # {'method':'EqOddsPostprocessing', 'params':{'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                            {'method':'CalibratedEqOddsPostprocessing_FPR', 'params':{'cost_constraint':'fpr','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                            {'method':'CalibratedEqOddsPostprocessing_FNR', 'params':{'cost_constraint':'fnr','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                            {'method':'CalibratedEqOddsPostprocessing_Weighted', 'params':{'cost_constraint':'weighted','unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                            {'method':'ThresholdOptimizer','params':{'estimator':base_estimator, 'constraints':'demographic_parity', 'objective':'accuracy_score', 'prefit':False}},
                            {'method':'ThresholdOptimizer_EqOdds','params':{'estimator':base_estimator, 'constraints':'equalized_odds', 'objective':'accuracy_score', 'prefit':False}},
                            {'method':'ThresholdOptimizer_FNR','params':{'estimator':base_estimator, 'constraints':'false_negative_rate_parity', 'objective':'accuracy_score', 'prefit':False}},
                            {'method':'ThresholdOptimizer_TPR','params':{'estimator':base_estimator, 'constraints':'true_positive_rate_parity', 'objective':'accuracy_score', 'prefit':False}},
                            {'method':'ThresholdOptimizer_FPR','params':{'estimator':base_estimator, 'constraints':'false_positive_rate_parity', 'objective':'accuracy_score', 'prefit':False}},
                            {'method':'ThresholdOptimizer_TNR','params':{'estimator':base_estimator, 'constraints':'true_negative_rate_parity', 'objective':'accuracy_score', 'prefit':False}}]
    
    method_lists = [pre_mitigation_methods, in_mitigation_methods, post_mitigation_methods]
    types = ['pre', 'in', 'post']

    mitigation_methods = [{'method': 'Baseline', 'params': {}, 'type':None}] + [
        {**d, 'type': t} for method_list, t in zip(method_lists, types) for d in method_list
    ] # list with baseline

    return mitigation_methods

def get_base_estimator(base_estimator_name):
    return RandomForestClassifier()

def load_attributes_covariance(data_name, state = None, year = None):
    if data_name == 'folktables':
        file_name = f'./results/stats/{data_name}/{str(year)}/{state}_attributes_stats.json'
    else:
        file_name = f'./results/stats/{data_name}/attributes_stats.json'

    # Check if the file exists
    if not os.path.exists(file_name):
        return None

    # File exists, proceed with opening and loading
    with open(file_name, 'r') as json_file:
        attributes_covariance = json.load(json_file)

    return attributes_covariance

def store_attributes_covariance(attributes_covariance, data_name, state = None, year = None):
    if 'folktables' in data_name:
        store_results_json(attributes_covariance, f'./results/stats/folktables/{str(year)}/{state}_attributes_stats.json')
    else:
        store_results_json(attributes_covariance, f'./results/stats/{data_name}/attributes_stats.json')

    return

def get_covariance_parameters(df, df_preprocessed, sensitive_attribute, alpha = None, target = 'Class', data_name = None, year = None, state = None):
    
    attributes_covariance = None

    if data_name != None:
        if 'folktables' in data_name and year != None and state != None:
            attributes_covariance = load_attributes_covariance('folktables',state, year)
        else:
            attributes_covariance = load_attributes_covariance(data_name)

    if attributes_covariance == None:
        print('Attributes stats not found in results. Calculating and storing...')
        attributes_covariance = get_attributes_covariance(df, sensitive_attribute)
        store_attributes_covariance(attributes_covariance, data_name, state, year)
    

    if alpha == None:
        alpha_covariance = get_alpha_covariance(df_preprocessed, attributes_covariance, sensitive_attribute, class_name=target, s_prediction_error=0.1, verbose=True)
    else:
        alpha_covariance = alpha

    return attributes_covariance, alpha_covariance

def get_methods_params(mitigation_methods, mitigation_method1, mitigation_method2):
    mitigation1, mitigation2 = None, None

    for method in mitigation_methods:
        if method['method'] == mitigation_method1:
            mitigation1 = method
        if method['method'] == mitigation_method2:
            mitigation2 = method
        
    return mitigation1, mitigation2

def get_methods(df, df_preprocessed, sensitive_attribute, mitigation_method1, mitigation_method2, experiment_type, privileged_group, unprivileged_group, base_estimator, target, data_name, alpha_covariance = None, state=None, year=None):
    
    if experiment_type == 'proxies' or 'Covariance' in mitigation_method1 or (mitigation_method2 != None and 'Covariance' in mitigation_method2):
        attributes_covariance, alpha_covariance = get_covariance_parameters(df, df_preprocessed, sensitive_attribute, alpha = alpha_covariance, target = target, 
                                                                            data_name=data_name, state=state, year=year)
    else:
        attributes_covariance = None

    mitigation_methods = get_presaved_methods(sensitive_attribute, base_estimator = base_estimator, 
                                              privileged_groups = privileged_group, unprivileged_groups = unprivileged_group, 
                                              attributes_covariance = attributes_covariance, alpha_covariance = alpha_covariance)
    
    mitigation1, mitigation2 = get_methods_params(mitigation_methods, mitigation_method1, mitigation_method2)

    return mitigation1, mitigation2

def store_results(results, experiment_type, data_name, sensitive_attribute, mitigation_method1, mitigation_method2):
    
    if experiment_type == 'hybrid':
        file_name = f'./results/{experiment_type}/{mitigation_method1}_{mitigation_method2}_{sensitive_attribute}_results_{data_name}.json'
    else:
        file_name = f'./results/{experiment_type}/{mitigation_method1}_{sensitive_attribute}_results_{data_name}.json'

    store_results_json(results, path=file_name)
    
    return


def main():
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Required arguments
    parser.add_argument('experiment_type', type=str, choices=['comparative', 'hybrid', 'proxies'], default='comparative', help='Method type (comparative, hybrid, proxies)')
    parser.add_argument('data_name', type=str, choices=['german', 'folktablesIncome'], help='Name of the data (german or folktablesIncome)')
    parser.add_argument('sensitive_attribute', type=str, help='Sensitive attribute')
    parser.add_argument('mitigation_method1', type=str, help='Mitigation method 1')
    
    # Optional arguments with default values
    parser.add_argument('--mitigation_method2', type=str, default=None, help='Mitigation method 2')
    parser.add_argument('--target', type=str, default='Class', help='Target')
    parser.add_argument('--favorable_class', type=int, default=1, help='Favorable class')
    parser.add_argument('--privileged_group', type=int, default=0, help='Privileged groups')
    parser.add_argument('--unfavorable_class', type=int, default=0, help='Unfavorable class')
    parser.add_argument('--unprivileged_group', type=int, default=1, help='Unprivileged groups')
    parser.add_argument('--base_estimator', type=str, default='random_forest', help='Base estimator')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--year', type=int, default=2018, help='Year (only for folktables)')
    parser.add_argument('--state', type=str, default='NY', help='State (only for folktables)')
    parser.add_argument('--alpha_covariance', type=float, default=None, help='Alpha for covariance method (If not provided it will be computed)')

    args = parser.parse_args()

    # Print the parsed values
    print(f"Experiment Type: {args.experiment_type}")
    print(f"Data Name: {args.data_name}")
    print(f"Sensitive Attribute: {args.sensitive_attribute}")
    print(f"Mitigation Method 1: {args.mitigation_method1}")
    print(f"Mitigation Method 2: {args.mitigation_method2}")
    print(f"Target: {args.target}")
    print(f"Favorable Class: {args.favorable_class}")
    print(f"Privileged Groups: {args.privileged_group}")
    print(f"Unfavorable Class: {args.unfavorable_class}")
    print(f"Unprivileged Groups: {args.unprivileged_group}")
    print(f"Base Estimator: {args.base_estimator}")
    print(f"Iterations: {args.iterations}")

    unprivileged_groups = [args.unprivileged_group] # Females
    privileged_groups = [args.privileged_group] # Males
    favorable_class = 1
    unfavorable_class = 0
    base_estimator = get_base_estimator(args.base_estimator)
    avg_results = {}

    # Get data
    if args.data_name == 'german':
        df, df_preprocessed = get_german_data()
    elif args.data_name == 'folktablesIncome':
        df, df_preprocessed = get_folktablesIncome_data(args.state, args.year)


    # Get methods
    mitigation1, mitigation2 = get_methods(df, df_preprocessed, args.sensitive_attribute, 
                                           args.mitigation_method1, args.mitigation_method2, args.experiment_type, 
                                           privileged_group=[args.privileged_group], unprivileged_group=[args.unprivileged_group], target = args.target,
                                           base_estimator=base_estimator, alpha_covariance=args.alpha_covariance,
                                           data_name=args.data_name, state=args.state, year=args.year)

    # Apply experiments
    if args.experiment_type == 'comparative':
        # Comparative
        performance_results, fairness_results = get_bias_mitigation_result(df_preprocessed, mitigation1, args.sensitive_attribute,
                                                                       target=args.target, favorable_class=favorable_class, privileged_groups=[args.privileged_group],
                                                                       iterations=args.iterations, base_estimator=base_estimator)
        pass
    elif args.experiment_type == 'hybrid':
        # Hyrbid
            
        performance_results, fairness_results = get_hybrid_bias_mitigation_result(df_preprocessed, args.sensitive_attribute, mitigation1, mitigation2, 
                                                                                      is_hybrid_exception = is_hybrid_exception, iterations = args.iterations, base_estimator=base_estimator,
                                                                                      target = args.target, favorable_class=args.favorable_class, privileged_groups = [args.privileged_group])
        pass
    elif args.experiment_type == 'proxies':
        #Proxies
        covariance_method, _ = get_methods(df, df_preprocessed, args.sensitive_attribute, 
                                           'Covariance', None, args.experiment_type, 
                                           privileged_group=[args.privileged_group], unprivileged_group=[args.unprivileged_group], target = args.target,
                                           base_estimator=base_estimator, alpha_covariance=args.alpha_covariance,
                                           data_name=args.data_name, state=args.state, year=args.year)
        performance_results, fairness_results = get_proxies_mitigation_results(df_preprocessed, args.sensitive_attribute, mitigation1, covariance_method, 
                                                                           iterations = args.iterations, target = args.target, 
                                                                           favorable_class=args.favorable_class, privileged_groups = [args.privileged_group], 
                                                                           base_estimator = base_estimator)
        pass

    
    if performance_results is not None and fairness_results is not None:
        for key in performance_results[0].keys():
            print(f'{key}: {sum(r[key] for r in performance_results) / len(performance_results)} ± {np.std([r[key] for r in performance_results])}')
            avg_results[key] = (sum(r[key] for r in performance_results) / len(performance_results), np.std([r[key] for r in performance_results]))

        for key in fairness_results[0].keys():
            print(f'{key}: {sum(r[key] for r in fairness_results) / len(fairness_results)} ± {np.std([r[key] for r in fairness_results])}')
            avg_results[key] = (sum(r[key] for r in fairness_results) / len(fairness_results), np.std([r[key] for r in fairness_results]))

    store_results(avg_results, args.experiment_type, args.data_name, args.sensitive_attribute, args.mitigation_method1, args.mitigation_method2)

if __name__ == "__main__":
    main()
