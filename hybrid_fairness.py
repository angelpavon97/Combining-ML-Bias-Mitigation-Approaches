
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from fairness_comparative import preprocessing_mitigation, inprocessing_mitigation, postprocessing_mitigation
from fairness_utils import test_model, set_new_estimator, performance_metrics 
import apply_metrics
from preprocessing import split_data, get_res_df

def hybrid_training(df, sensitive_attribute, pre_mitigation = None, in_mitigation = None, post_mitigation = None, iter = 10, target = 'Class', favorable_class=1, privileged_groups = [0], base_estimator = RandomForestClassifier()):
    performance_results = []
    fairness_results = []

    # If there is a in-mitigation type we need different testing
    if in_mitigation != None:
        mitigation_type = 'in'
    else:
        mitigation_type = None

    for i in range(iter):
        # train/test split
        X_train, X_test, y_train, y_test = split_data(df, test_size = 0.2, seed = i)
        # save test sensitive attribute column
        s_train = X_train[sensitive_attribute]
        s_test = X_test[sensitive_attribute]

        # pre-processing mitigation
        if pre_mitigation != None:  
            X_train, y_train, X_test, y_test, train_weights = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                    method = pre_mitigation['method'], 
                                                                    params = pre_mitigation['params'],
                                                                    seed = i,
                                                                    favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        else:
            # Remove sensitive attribute before training
            X_train, y_train, X_test, y_test, train_weights = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                    method = 'Unawareness', 
                                                                    params = None,
                                                                    favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
            
            
        # in-processing mitigation    
        if in_mitigation != None:

            # Get the sensitive attribute back as in-processing methods needs it
            X_train[sensitive_attribute] = s_train
            X_test[sensitive_attribute] = s_test

            clf = inprocessing_mitigation(X_train, y_train, sensitive_attribute,
                                        method = in_mitigation['method'], 
                                        params = in_mitigation['params'],
                                        train_weights = train_weights,
                                        seed = i,
                                        base_estimator = base_estimator, favorable_class=favorable_class, privileged_groups = privileged_groups)
        else:
            clf = inprocessing_mitigation(X_train, y_train, sensitive_attribute,
                                        method = None, 
                                        params = None,
                                        train_weights = train_weights,
                                        base_estimator = base_estimator, favorable_class=favorable_class, privileged_groups = privileged_groups)

        # test model (get model predictions)
        y_pred_test = test_model(clf, X_test, mitigation_type = mitigation_type, y_test = y_test, sensitive_attribute = sensitive_attribute,
                                  favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        y_pred_train = test_model(clf, X_train, mitigation_type = mitigation_type, y_test = y_train, sensitive_attribute = sensitive_attribute,
                                  favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        # post-processing mitigation
        if post_mitigation != None:
            
            # Get the sensitive attribute back as post-processing methods needs it
            X_train[sensitive_attribute] = s_train
            X_test[sensitive_attribute] = s_test

            # If a postprocessing uses a base estimator, change it for the one already trained
            post_mitigation['params'] = set_new_estimator(post_mitigation['params'], clf)
            
            post_y_pred = postprocessing_mitigation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, sensitive_attribute,
                                                method = post_mitigation['method'], 
                                                params = post_mitigation['params'],
                                                seed = i,
                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        else:
            post_y_pred = postprocessing_mitigation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, sensitive_attribute,
                                                method = None, 
                                                params = None,
                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
            
        # get results
        performance_results.append(performance_metrics(y_test, post_y_pred))

        res_df = get_res_df(X_test, y_test, post_y_pred)
        fairness_results.append(apply_metrics.fair_metrics(res_df))
    
    return performance_results, fairness_results

def get_mitigation_combination(mitigation1, mitigation2):

    pre_mitigation = None
    in_mitigation = None
    post_mitigation = None

    if mitigation1['type'] == 'pre':
        pre_mitigation = mitigation1
    elif mitigation1['type'] == 'in':
        in_mitigation = mitigation1
    elif mitigation1['type'] == 'post':
        post_mitigation = mitigation1

    if mitigation2['type'] == 'pre':
        pre_mitigation = mitigation2
    elif mitigation2['type'] == 'in':
        in_mitigation = mitigation2
    elif mitigation2['type'] == 'post':
        post_mitigation = mitigation2

    return pre_mitigation, in_mitigation, post_mitigation

def get_hybrid_bias_mitigation_result(df, sensitive_attribute, mitigation1, mitigation2, is_hybrid_exception = None, iterations = 10, target = 'Class', favorable_class=1, privileged_groups = [0], base_estimator = RandomForestClassifier()):
    if is_hybrid_exception is not None:
        if is_hybrid_exception(mitigation1, mitigation2):
            return None, None
        
    if mitigation1['type'] == mitigation2['type']:
        print('Warning: Same type hybrid mitigation is not allowed')
        return None
    
    print(f"\nMitigation 1: {mitigation1['method']} - Mitigation 2: {mitigation2['method']}")
    print(f"Mitigation Type 1: {mitigation1['type']} - Mitigation Type 2: {mitigation2['type']}")
    
    pre_mitigation, in_mitigation, post_mitigation = get_mitigation_combination(mitigation1, mitigation2)
    
    performance_results, fairness_results = hybrid_training(df, sensitive_attribute, pre_mitigation, in_mitigation, post_mitigation, iter = iterations, 
                    target = target, favorable_class=favorable_class, privileged_groups = privileged_groups, base_estimator=base_estimator)
    
    return performance_results, fairness_results


def get_proxies_mitigation_results(df, sensitive_attribute, mitigation, covariance_method, iterations = 10, target = 'Class', favorable_class=1, privileged_groups = [0], base_estimator = RandomForestClassifier()):
    performance_results = []
    fairness_results = []
    
    for i in range(iterations):
        # train/test split
        X_train, X_test, y_train, y_test = split_data(df, test_size = 0.2, seed = i)

        # Apply covariance method
        X_train_cov, y_train_cov, X_test_cov, y_test_cov, _ = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                method = covariance_method['method'], 
                                                                params = covariance_method['params'],
                                                                drop_sensitive_attribute = False,
                                                                seed = i,
                                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        # Get the subset of attributes considered proxies by Covariance method
        X_train_proxies = X_train[X_train.columns.difference(X_train_cov.columns)]
        X_test_proxies = X_test[X_test.columns.difference(X_test_cov.columns)]

        # Pre-processing mitigation methods just in the proxies
        X_train_trans, y_train_trans, X_test_trans, y_test_trans, train_weights = preprocessing_mitigation(X_train_proxies, y_train, X_test_proxies, y_test, sensitive_attribute,
                                                                method = mitigation['method'], 
                                                                params = mitigation['params'],
                                                                drop_sensitive_attribute = False,
                                                                seed = i,
                                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        # Concat pre-processed proxies with the rest of data not transformed
        X_train_trans = pd.concat([X_train_trans, X_train_cov], axis=1)
        X_test_trans = pd.concat([X_test_trans, X_test_cov], axis=1)

        # Remove sensitive attribute from train and test before training and prediction
        if sensitive_attribute in X_train_trans:
            X_train_trans = X_train_trans.drop(sensitive_attribute, axis=1)
        
        if sensitive_attribute in X_test_trans:
            X_test_trans = X_test_trans.drop(sensitive_attribute, axis=1)

        # Train model (method = None)
        clf = inprocessing_mitigation(X_train_trans, y_train_trans, sensitive_attribute,
                                        method = None, 
                                        params = None,
                                        train_weights=train_weights,
                                        base_estimator = base_estimator, favorable_class=favorable_class, privileged_groups = privileged_groups) 
        
        # test model (get model predictions)
        y_pred_test = test_model(clf, X_test_trans, mitigation_type = mitigation['type'], y_test = y_test, sensitive_attribute = sensitive_attribute,
                                 favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        y_pred_train = test_model(clf, X_train_trans, mitigation_type = mitigation['type'], y_test = y_train, sensitive_attribute = sensitive_attribute,
                                 favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        post_y_pred = postprocessing_mitigation(X_train_trans, X_test_trans, y_train_trans, y_test, y_pred_train, y_pred_test, sensitive_attribute, 
                                                method = None, 
                                                params = None,
                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        
        # get results
        performance_results.append(performance_metrics(y_test, post_y_pred))

        X_test_trans[sensitive_attribute] = X_test[sensitive_attribute] # Get the sensitive attribute back for fairness evaluation
        res_df = get_res_df(X_test_trans, y_test, post_y_pred)
        fairness_results.append(apply_metrics.fair_metrics(res_df))

        return performance_results, fairness_results