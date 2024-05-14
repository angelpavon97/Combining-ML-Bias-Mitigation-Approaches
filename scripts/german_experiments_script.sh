#!/bin/bash

commands=(
	"python3 main.py comparative german gender Baseline --iterations 10"
	"python3 main.py comparative german gender Unawareness --iterations 10"
	"python3 main.py comparative german gender DisparateImpactRemover --iterations 10"
	"python3 main.py comparative german gender Reweighing --iterations 10"
	"python3 main.py comparative german gender LFR --iterations 10"
	"python3 main.py comparative german gender Covariance --alpha_covariance 0.01 --iterations 10"
	"python3 main.py comparative german gender Covariance05 --alpha_covariance 0.01 --iterations 10"
	"python3 main.py comparative german gender Covariance09 --alpha_covariance 0.01 --iterations 10"
	"python3 main.py comparative german gender CorrelationRemover --iterations 10"
	"python3 main.py comparative german gender GerryFairClassifier_FP --iterations 10"
	"python3 main.py comparative german gender GerryFairClassifier_FN --iterations 10"
	"python3 main.py comparative german gender ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py comparative german gender ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py comparative german gender ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py comparative german gender ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py comparative german gender MetaFairClassifier_DI --iterations 10"
	"python3 main.py comparative german gender MetaFairClassifier_FDR --iterations 10"
	"python3 main.py comparative german gender AdversialDebiasing --iterations 10"
	"python3 main.py comparative german gender RejectOptionClassification_SP --iterations 10"
	"python3 main.py comparative german gender RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py comparative german gender RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py comparative german gender CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py comparative german gender CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py comparative german gender CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py comparative german gender ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender Unawareness --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender Reweighing --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
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
	"python3 main.py hybrid german gender Covariance --mitigation_method2 GerryFairClassifier_FP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 GerryFairClassifier_FN --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ExponentiatedGradientReduction_DP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ExponentiatedGradientReduction_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ExponentiatedGradientReduction_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ExponentiatedGradientReduction_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 MetaFairClassifier_DI --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 MetaFairClassifier_FDR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 AdversialDebiasing --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 RejectOptionClassification_SP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 RejectOptionClassification_AvgOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 RejectOptionClassification_EqOpp --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance --mitigation_method2 ThresholdOptimizer_TNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 GerryFairClassifier_FP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 GerryFairClassifier_FN --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ExponentiatedGradientReduction_DP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ExponentiatedGradientReduction_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ExponentiatedGradientReduction_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ExponentiatedGradientReduction_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 MetaFairClassifier_DI --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 MetaFairClassifier_FDR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 AdversialDebiasing --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 RejectOptionClassification_SP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 RejectOptionClassification_AvgOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 RejectOptionClassification_EqOpp --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance05 --mitigation_method2 ThresholdOptimizer_TNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 GerryFairClassifier_FP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 GerryFairClassifier_FN --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ExponentiatedGradientReduction_DP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ExponentiatedGradientReduction_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ExponentiatedGradientReduction_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ExponentiatedGradientReduction_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 MetaFairClassifier_DI --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 MetaFairClassifier_FDR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 AdversialDebiasing --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 RejectOptionClassification_SP --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 RejectOptionClassification_AvgOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 RejectOptionClassification_EqOpp --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer_EqOdds --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer_FNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer_TPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer_FPR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender Covariance09 --mitigation_method2 ThresholdOptimizer_TNR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 GerryFairClassifier_FP --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 GerryFairClassifier_FN --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ExponentiatedGradientReduction_DP --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ExponentiatedGradientReduction_EqOdds --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ExponentiatedGradientReduction_TPR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ExponentiatedGradientReduction_FPR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 MetaFairClassifier_DI --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 MetaFairClassifier_FDR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 AdversialDebiasing --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender CorrelationRemover --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FP --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender GerryFairClassifier_FN --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_DP --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_EqOdds --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_TPR --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender ExponentiatedGradientReduction_FPR --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_DI --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender MetaFairClassifier_FDR --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 RejectOptionClassification_SP --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 RejectOptionClassification_AvgOdds --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 RejectOptionClassification_EqOpp --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 CalibratedEqOddsPostprocessing_FPR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 CalibratedEqOddsPostprocessing_FNR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 CalibratedEqOddsPostprocessing_Weighted --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer_FNR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer_TPR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer_FPR --iterations 10"
	"python3 main.py hybrid german gender AdversialDebiasing --mitigation_method2 ThresholdOptimizer_TNR --iterations 10"
	"python3 main.py proxies german gender Unawareness --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies german gender DisparateImpactRemover --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies german gender Reweighing --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies german gender LFR --alpha_covariance 0.01 --iterations 10"
	"python3 main.py proxies german gender CorrelationRemover --alpha_covariance 0.01 --iterations 10"
)

# Number of commands to run concurrently
max_concurrent=10

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
    