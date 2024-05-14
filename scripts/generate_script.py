
from sklearn.ensemble import RandomForestClassifier
import argparse
def get_presaved_methods(sensitive_attribute, base_estimator = RandomForestClassifier(), privileged_groups = [0], unprivileged_groups = [1], attributes_covariance = None, alpha_covariance = None):
    pre_mitigation_methods = [{'method':'Unawareness', 'params':{}},
                          {'method':'DisparateImpactRemover', 'params':{'sensitive_attribute':sensitive_attribute, 'repair_level':0.5}},
                          {'method':'Reweighing', 'params':{'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}},
                          {'method':'LFR', 'params':{'k':5, 'Ax':1.0, 'Ay':1.0, 'Az':1.0,'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}, # New
                          {'method':'LFRDefault', 'params':{'k':5, 'Ax':0.01, 'Ay':1.0, 'Az':50.0,'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}, # New
                          {'method':'Covariance', 'params':{'attributes_covariance':attributes_covariance, 'alpha':alpha_covariance}},
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
                         {'method':'AdversialDebiasing', 'params':{'scope_name':'classifier','sess':None,'unprivileged_groups':[{sensitive_attribute: unprivileged_groups[0]}] , 'privileged_groups':[{sensitive_attribute: privileged_groups[0]}]}}]
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

def main():
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Required arguments
    parser.add_argument('data_name', type=str, choices=['german', 'folktablesIncome'], help='Name of the data (german or folktablesIncome)')
    parser.add_argument('sensitive_attribute', type=str, help='Sensitive attribute')
    parser.add_argument('--alpha_covariance', type=float, default=None, help='Alpha for covariance method (If not provided it will be computed)')
    parser.add_argument('--iterations', type=int, default=10, help='Iterations per experiment')

    args = parser.parse_args()

    mitigation_methods = get_presaved_methods(sensitive_attribute='gender')
    f = open(f"{args.data_name}_experiments_script.sh", "w")
    f.write('#!/bin/bash\n\ncommands=(\n')

    for m in mitigation_methods:
        if 'Covariance' not in m['method'] or args.alpha_covariance is None:
            f.write(f'\t"python3 main.py comparative {args.data_name} {args.sensitive_attribute} {m["method"]} --iterations {args.iterations}"\n')
        else:
            f.write(f'\t"python3 main.py comparative {args.data_name} {args.sensitive_attribute} {m["method"]} --alpha_covariance {args.alpha_covariance} --iterations {args.iterations}"\n')
    
    combinations_done = {}
    for mitigation1 in mitigation_methods:
        combinations_done[mitigation1['method']] = {}
        for mitigation2 in mitigation_methods:
            if mitigation1['type'] == mitigation2['type'] or mitigation1['method'] == 'Baseline' or mitigation2['method'] == 'Baseline':
                pass
            elif mitigation2['method'] in combinations_done and mitigation1['method'] in combinations_done[mitigation2['method']]:
                # Check if mititigation2 and mitigation1 have been done in previous iteration
                combinations_done[mitigation1['method']][mitigation2['method']] = combinations_done[mitigation2['method']][mitigation1['method']]
                pass
            else:
                if ('Covariance' in mitigation1['method'] or 'Covariance' in mitigation2['method']) and args.alpha_covariance is not None:
                    f.write(f'\t"python3 main.py hybrid {args.data_name} {args.sensitive_attribute} {mitigation1["method"]} --mitigation_method2 {mitigation2["method"]} --alpha_covariance {args.alpha_covariance} --iterations {args.iterations}"\n')
                else:
                    f.write(f'\t"python3 main.py hybrid {args.data_name} {args.sensitive_attribute} {mitigation1["method"]} --mitigation_method2 {mitigation2["method"]} --iterations {args.iterations}"\n')
                    
                combinations_done[mitigation1['method']][mitigation2['method']] = True

    for m in mitigation_methods:
        if m['type'] == 'pre' and 'Covariance' not in m['method']:
            f.write(f'\t"python3 main.py proxies {args.data_name} {args.sensitive_attribute} {m["method"]} --alpha_covariance {args.alpha_covariance} --iterations {args.iterations}"\n')

    script_content = """
# Number of commands to run concurrently
max_concurrent=5

run_commands() {
    pids=()
    
    for command in "${commands[@]}"; do
        $command &  # Execute the command in the background
        pids+=($!)  # Store the process ID (PID) of the last background command

        if (( ${#pids[@]} >= max_concurrent )); then
            for pid in "${pids[@]}"; do
                wait $pid  # Wait for each background process to finish
            done
            pids=()  # Clear the array for the next batch of commands
        fi
    done

    for pid in "${pids[@]}"; do
        wait $pid  # Wait for any remaining background processes to finish
    done
}

# Run the commands
run_commands
    """

    f.write(f')\n{script_content}')
    f.close()

if __name__ == "__main__":
    main()