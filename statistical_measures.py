
from scipy.stats import ttest_ind, chi2_contingency, entropy, kruskal, mannwhitneyu, fisher_exact
from sklearn import metrics
import pandas as pd
import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

def get_entropy(s):
    return entropy(s.value_counts(normalize=True))

def get_mi(df, class_name = 'Class'):
    
    mis = {}
    
    for col_name in df.select_dtypes(object).columns:
        mis[col_name] = metrics.mutual_info_score(df[class_name], df[col_name])

    return {k: v for k, v in sorted(mis.items(), key=lambda item: item[1], reverse=True)}

def get_gr(df, class_name = 'Class'):
    
    grs = {}
    
    for col_name in df.select_dtypes(object).columns:
        mi = metrics.mutual_info_score(df[class_name], df[col_name])
        grs[col_name] = mi/get_entropy(df[col_name])

    return {k: v for k, v in sorted(grs.items(), key=lambda item: item[1], reverse=True)}

def get_suc(df, class_name = 'Class'):
    
    sucs = {}
    
    for col_name in df.select_dtypes(object).columns:
        mi = metrics.mutual_info_score(df[class_name], df[col_name])
        sucs[col_name] = 2 * (mi/(get_entropy(df[col_name]) + get_entropy(df[class_name])))

    return {k: v for k, v in sorted(sucs.items(), key=lambda item: item[1], reverse=True)}

def check_chi2_assumptions(contingency, col_name, verbose = True):
    
    # all individual expected counts are 1 or greater
    if contingency.equals(contingency[contingency >= 1]) == False:
        # print(f'WARNING: Assumption is not met in {col_name} contingency table as some expected counts are less than 1.')
        col_values = contingency[contingency < 1].columns.values
        if verbose:
            print(f'WARNING: The assumption of all individual expected cell counts being 1 or greater is not met in column "{col_name}" at value(s) {col_values}.')
        
        return -1
    
    # No more than 20% of the expected counts are less than 5
    n_true = contingency[contingency >= 5].count().sum()
    n_false = contingency[contingency < 5].count().sum()
    
    if n_false/(n_true + n_false) > 0.2:
        if verbose:
            print(f'WARNING: Assumption is not met in {col_name} contingency table as more than 20% of expected counts are less than 5.')
        
        return -2
    
    return 0

def get_chi2(df, class_name = 'Class', alpha = 0.05, strict = False): 
    dependent_attributes = {}
    independent_attributes = {}
    
    for col_name in df.select_dtypes(object).columns:
        contingency = pd.crosstab(df[class_name], df[col_name])

        warn_code = check_chi2_assumptions(contingency, col_name, verbose = not(strict))

        if strict == False or warn_code == 0:
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            if p_value < alpha:
                dependent_attributes[col_name] = p_value
            else:
                independent_attributes[col_name] = p_value

    d_sorted = {k: v for k, v in sorted(dependent_attributes.items(), key=lambda item: item[1], reverse=False)}
    i_sorted = {k: v for k, v in sorted(independent_attributes.items(), key=lambda item: item[1], reverse=False)}
    
    return d_sorted, i_sorted

def get_fisher_exact(df, class_name = 'Class', alpha = 0.05, verbose = True):
    
    dependent_attributes = {}
    independent_attributes = {}
    
    for col_name in df.select_dtypes(object).columns:
        contingency = pd.crosstab(df[class_name], df[col_name])

        warn_code = check_chi2_assumptions(contingency, col_name, verbose = False)
        
        if warn_code < 0:
            #  statistic, p_value = fisher_exact(contingency, alternative='two-sided')
            rpy2.robjects.numpy2ri.activate()
            r_stats = importr('stats')
            try:
                p_value = r_stats.fisher_test(contingency.to_numpy())[0][0]
            except:
                if verbose == True:
                    print(f'Warning: Simulating p-value for {col_name} due an error')
                p_value = r_stats.fisher_test(contingency.to_numpy(), simulate_p_value=True)[0][0]
            
            if p_value < alpha:
                dependent_attributes[col_name] = p_value
            else:
                independent_attributes[col_name] = p_value

    d_sorted = {k: v for k, v in sorted(dependent_attributes.items(), key=lambda item: item[1], reverse=False)}
    i_sorted = {k: v for k, v in sorted(independent_attributes.items(), key=lambda item: item[1], reverse=False)}
    
    return d_sorted, i_sorted

def combine_dicts(list_of_dicts, sort = False):
    result_dict = {}
    
    for d in list_of_dicts:
        result_dict.update(d)
    
    if sort == True:
        result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=False)}
    
    return result_dict

def apply_categorical_tests(df, class_name = 'Class', alpha = 0.05, verbose = True):
    dep_fisher, ind_fisher = get_fisher_exact(df, class_name = class_name, alpha = alpha, verbose = verbose)
    dep_chi2, ind_chi2 = get_chi2(df, class_name = class_name, alpha = alpha, strict = True)
    
    return combine_dicts([dep_fisher, dep_chi2], sort = True), combine_dicts([ind_fisher, ind_chi2], sort = True)


def get_mannwhitneyu(df, class_name = 'Class', alpha = 0.05, nan_policy = 'omit'):
    
    diff_dis_attributes = {}
    same_dis_attributes = {}
    
    # Separate both populations (ex: if class_name = gender, df1 will be for females and df2 for males)
    class_values = list(set(df[class_name].values))
    df1 = df[df[class_name] == class_values[0]]
    df2 = df[df[class_name] != class_values[0]]
    
    for col_name in df.select_dtypes([np.number]).columns:
        t_statistic, p_value = mannwhitneyu(df1[col_name], df2[col_name], nan_policy=nan_policy)

        if p_value < alpha:
            diff_dis_attributes[col_name] = p_value
        else:
            same_dis_attributes[col_name] = p_value
            
    d_sorted = {k: v for k, v in sorted(diff_dis_attributes.items(), key=lambda item: item[1], reverse=False)}
    s_sorted = {k: v for k, v in sorted(same_dis_attributes.items(), key=lambda item: item[1], reverse=False)}
    
    return d_sorted, s_sorted