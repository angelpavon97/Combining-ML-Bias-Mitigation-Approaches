import pandas as pd

# Pre-processing
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing, LFR
from fairlearn.preprocessing import CorrelationRemover
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
# In-processing
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from aif360.algorithms.inprocessing import AdversarialDebiasing, ARTClassifier, GerryFairClassifier, MetaFairClassifier, PrejudiceRemover, ExponentiatedGradientReduction, GridSearchReduction
# Post-processing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from fairlearn.postprocessing import ThresholdOptimizer

from fairness_utils import convert_X_to_aif360, convert_aif360_to_X, covariance_mitigation, set_new_estimator, performance_metrics, test_model
import apply_metrics
from preprocessing import split_data, get_res_df


def preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute, method = None, params = {}, seed = None, drop_sensitive_attribute = True, favorable_class=1, privileged_groups = [0], target = 'Class'):

    aif360_df_train = convert_X_to_aif360(X_train, y_train, sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)
    aif360_df_test = convert_X_to_aif360(X_test, y_test, sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)

    if method == None or method =='Baseline':
        return X_train, y_train, X_test, y_test, aif360_df_train.instance_weights

    elif 'Unawareness' in method:
        if sensitive_attribute in X_train:
            X_train = X_train.drop(sensitive_attribute, axis=1)
        
        if sensitive_attribute in X_test:
            X_test = X_test.drop(sensitive_attribute, axis=1)
            
        train_weights = aif360_df_train.instance_weights   
        return X_train, y_train, X_test, y_test, train_weights
    
    elif 'DisparateImpactRemover' in method:
        preprocessor = DisparateImpactRemover(**params)
        aif360_df_processed_train = preprocessor.fit_transform(aif360_df_train)
        # aif360_df_processed_test = preprocessor.transform(aif360_df_test) transform not allowed
        aif360_df_processed_test = aif360_df_test

    elif 'LFR' in method:
        preprocessor = LFR(**params)
        aif360_df_processed_train = preprocessor.fit_transform(aif360_df_train, maxiter=5000, maxfun=5000)
        aif360_df_processed_test = preprocessor.transform(aif360_df_test)
    
    elif 'Reweighing' in method:
        preprocessor = Reweighing(**params)
        aif360_df_processed_train = preprocessor.fit_transform(aif360_df_train)
        aif360_df_processed_test = preprocessor.transform(aif360_df_test)

    elif 'OptimPreproc' in method:
        # preprocessor = OptimPreproc(**params)
        # aif360_df_processed = preprocessor.fit_transform(aif360_df)
        aif360_df_train = convert_X_to_aif360(X_train.astype(float), y_train.astype(float), sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)

        optim_options = {}
    
        OP = OptimPreproc(OptTools, optim_options)

        OP = OP.fit(aif360_df_train)
        aif360_df_processed_train = OP.transform(aif360_df_train, transform_Y=True)
        aif360_df_processed_train = aif360_df_train.align_datasets(aif360_df_processed_train)

        aif360_df_processed_test = OP.transform(aif360_df_test, transform_Y=False)
        aif360_df_processed_test = aif360_df_test.align_datasets(aif360_df_processed_test)

    elif 'Covariance' in method:
        X_train = covariance_mitigation(X_train, attributes_covariance = params['attributes_covariance'], alpha = params['alpha'])
        X_test = X_test[X_train.columns]
        train_weights = aif360_df_train.instance_weights    
        return X_train, y_train, X_test, y_test, train_weights
    
    elif 'CorrelationRemover' in method:
        cr = CorrelationRemover(**params)
        cr.fit(X_train)

        X_train_processed = cr.transform(X_train)
        X_test_processed = cr.transform(X_test)

        # Get the columns excluding the sensitive attribute, as cr removes the sensitive attribute
        columns_to_keep = [col for col in X_train.columns if col != sensitive_attribute]

        # Convert the processed arrays back to DataFrames
        X_train_processed = pd.DataFrame(X_train_processed, index=X_train.index, columns=columns_to_keep)
        X_test_processed = pd.DataFrame(X_test_processed, index=X_test.index, columns=columns_to_keep)

        train_weights = aif360_df_train.instance_weights

        return X_train_processed, y_train, X_test_processed, y_test, train_weights

    X_train, y_train = convert_aif360_to_X(aif360_df_processed_train, target = target)
    X_test, _ = convert_aif360_to_X(aif360_df_processed_test, target = target) # we do not want the y_test transformed
    train_weights = aif360_df_processed_train.instance_weights

    if drop_sensitive_attribute == True:
        if sensitive_attribute in X_train:
            X_train = X_train.drop(sensitive_attribute, axis=1)
        
        if sensitive_attribute in X_test:
            X_test = X_test.drop(sensitive_attribute, axis=1)

    return X_train, y_train, X_test, y_test, train_weights


def inprocessing_mitigation(X, y, sensitive_attribute, method = None, params = {}, seed = None, train_weights = None, base_estimator = RandomForestClassifier(), favorable_class=1, privileged_groups = [0]):

    if method == None or method =='Baseline':
        # clf = XGBClassifier(seed = seed)
        clf = base_estimator
        
        clf.fit(X,y, sample_weight=train_weights)

        return clf

    aif360_df = convert_X_to_aif360(X, y.astype('int64'), sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)
    aif360_df.instance_weights = train_weights

    if 'AdversialDebiasing' in method:
        tf.compat.v1.disable_eager_execution()
        params['scope_name'] = params['scope_name'] + str(seed) # Every iteration will be a different name so they not overlap
        debias_model = AdversarialDebiasing(**params)
        debias_model.fit(aif360_df)

    elif 'GerryFairClassifier' in method:
        debias_model = GerryFairClassifier(**params)
        debias_model.fit(aif360_df)

    elif 'MetaFairClassifier' in method:
        debias_model = MetaFairClassifier(**params, seed=seed)
        debias_model.fit(aif360_df)

    elif 'PrejudiceRemover' in method:
        debias_model = PrejudiceRemover(**params)
        debias_model.fit(aif360_df)

    elif 'ExponentiatedGradientReduction' in method:
        debias_model = ExponentiatedGradientReduction(**params)
        debias_model.fit(aif360_df)

    elif 'GridSearchReduction' in method:
        debias_model = GridSearchReduction(**params)
        debias_model.fit(aif360_df)

    return debias_model


def postprocessing_mitigation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, sensitive_attribute, method = None, params = {}, seed = None, target = 'Class', favorable_class=1, privileged_groups = [0]):

    if method == None or method =='Baseline':
        return y_pred_test
    
    elif 'ThresholdOptimizer' in method: # Fairlearn method
        post_method = ThresholdOptimizer(**params)
        post_method.fit(X_train.drop(sensitive_attribute, axis=1), 
                        y_train, 
                        sensitive_features=X_train[sensitive_attribute])
        post_y_pred = post_method.predict(X_test.drop(sensitive_attribute, axis=1), 
                                          sensitive_features=X_test[sensitive_attribute],
                                          random_state=seed)
        return list(post_y_pred)

    y_pred_train = pd.Series(y_pred_train, index=y_train.index)
    y_pred_test = pd.Series(y_pred_test, index=y_test.index)

    y_pred_train = y_pred_train.values
    y_pred_test = y_pred_test.values

    aif360_df_train = convert_X_to_aif360(X_train, y_train, sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)
    aif360_df_train_pred = aif360_df_train.copy(deepcopy=True)
    aif360_df_train_pred.scores = y_pred_train.reshape(-1,1)
    aif360_df_train_pred.labels = y_train.values.reshape(-1,1)

    aif360_df_test = convert_X_to_aif360(X_test, y_test, sensitive_attribute,
                                          favorable_class=favorable_class, privileged_groups=privileged_groups)
    aif360_df_test_pred = aif360_df_test.copy(deepcopy=True)
    aif360_df_test_pred.scores = y_pred_test.reshape(-1,1)
    aif360_df_test_pred.labels = y_test.values.reshape(-1,1)

    if 'CalibratedEqOddsPostprocessing' in method:
        post_method = CalibratedEqOddsPostprocessing(**params)
        post_method.fit(aif360_df_train, aif360_df_train_pred)

        aif360_df_res = post_method.predict(aif360_df_test_pred)
        X_res, y_res = convert_aif360_to_X(aif360_df_res, target=target)
        post_y_pred = list(y_res)

    elif 'EqOddsPostprocessing' in method:
        post_method = EqOddsPostprocessing(**params)
        post_method.fit(aif360_df_train, aif360_df_train_pred)

        aif360_df_res = post_method.predict(aif360_df_test_pred)
        X_res, y_res = convert_aif360_to_X(aif360_df_res, target=target)
        post_y_pred = list(y_res)

    elif 'RejectOptionClassification' in method:
        post_method = RejectOptionClassification(**params)
        post_method.fit(aif360_df_train, aif360_df_train_pred)

        aif360_df_res = post_method.predict(aif360_df_test_pred)
        X_res, y_res = convert_aif360_to_X(aif360_df_res, target=target)
        post_y_pred = list(y_res)

    return post_y_pred


def get_bias_mitigation_result(df, mitigation, sensitive_attribute, target = 'Class', favorable_class=1, privileged_groups = [0], iterations = 10, base_estimator = RandomForestClassifier()):

    print(f"\nMitigation: {mitigation['method']} - Mitigation Type: {mitigation['type']}")
    performance_results = []
    fairness_results = []
    
    
    for i in range(iterations):
        # train/test split
        X_train, X_test, y_train, y_test = split_data(df, test_size = 0.2, seed = i)

        # save sensitive attribute column
        s_train = X_train[sensitive_attribute]
        s_test = X_test[sensitive_attribute]

        # pre-processing mitigation
        if mitigation['type'] == 'pre': 
            X_train, y_train, X_test, y_test, train_weights = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                    method = mitigation['method'], 
                                                                    params = mitigation['params'],
                                                                    seed = i,
                                                                    favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        elif mitigation['type'] != 'in' and mitigation['method'] != 'Baseline':
            # Remove sensitive attribute before training
            X_train, y_train, X_test, y_test, train_weights = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                    method = 'Unawareness', 
                                                                    params = None,
                                                                    favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        else:
            # If in-processing or baseline, we don't remove sensitive attribute as it is needed or not taken into account
             X_train, y_train, X_test, y_test, train_weights = preprocessing_mitigation(X_train, y_train, X_test, y_test, sensitive_attribute,
                                                                    method = None, 
                                                                    params = None,
                                                                    favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
            
        # in-processing mitigation    
        if mitigation['type'] == 'in':
            clf = inprocessing_mitigation(X_train, y_train, sensitive_attribute,
                                        method = mitigation['method'], 
                                        params = mitigation['params'],
                                        train_weights = train_weights,
                                        seed = i,
                                        base_estimator = base_estimator, favorable_class=favorable_class, privileged_groups = privileged_groups)
        else:
            # Training
            clf = inprocessing_mitigation(X_train, y_train, sensitive_attribute,
                                        method = None, 
                                        params = None,
                                        train_weights = train_weights,
                                        base_estimator = base_estimator, favorable_class=favorable_class, privileged_groups = privileged_groups) 
        
        # test model (get model predictions)
        y_pred_test = test_model(clf, X_test, mitigation_type = mitigation['type'], y_test = y_test, sensitive_attribute = sensitive_attribute,
                                 favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        y_pred_train = test_model(clf, X_train, mitigation_type = mitigation['type'], y_test = y_train, sensitive_attribute = sensitive_attribute,
                                  favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)

        # post-processing mitigation
        if mitigation['type'] == 'post':
            
            # Get the sensitive attribute for post processing mitigation
            X_train[sensitive_attribute] = s_train
            X_test[sensitive_attribute] = s_test

            # If a postprocessing uses a base estimator, change it for the one already trained
            mitigation['params'] = set_new_estimator(mitigation['params'], clf)

            post_y_pred = postprocessing_mitigation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, sensitive_attribute, 
                                                method = mitigation['method'], 
                                                params = mitigation['params'],
                                                seed = i,
                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        else:
            post_y_pred = postprocessing_mitigation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, sensitive_attribute, 
                                                method = None, 
                                                params = None,
                                                favorable_class=favorable_class, privileged_groups = privileged_groups, target = target)
        
        # get results
        performance_results.append(performance_metrics(y_test, post_y_pred))
        X_test[sensitive_attribute] = s_test
        res_df = get_res_df(X_test, y_test, post_y_pred)
        fairness_results.append(apply_metrics.fair_metrics(res_df))

    return performance_results, fairness_results