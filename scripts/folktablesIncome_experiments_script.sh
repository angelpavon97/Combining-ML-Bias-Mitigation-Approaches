#!/bin/bash

commands=(
	"python3 main.py comparative german gender LFR --iterations 10"
	"python3 main.py comparative german gender LFRDefault --iterations 10"
	"python3 main.py comparative folktablesIncome gender LFR --iterations 10"
	"python3 main.py comparative folktablesIncome gender LFRDefault --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender LFR --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender LFRDefault --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py proxies german gender LFR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies german gender LFRDefault --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFR --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid folktablesIncome gender LFRDefault --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py proxies folktablesIncome gender LFR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies folktablesIncome gender LFRDefault --alpha_covariance 0.01 --iterations 10"
)

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
    