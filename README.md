# Combining Machine Learning Bias Mitigation Approaches to Satisfy Multiple Fairness Constraints

This repository contains supplementary materials related to the paper titled "Combining Machine Learning Bias Mitigation Approaches to Satisfy Multiple Fairness Constraints" (under review). The paper addresses the challenge of mitigating bias in machine learning algorithms in multiple fairness metrics. The proposed approach combines existing bias mitigation approaches targeting different fairness metrics to mitigate bias in multiple fairness metrics at the same time.

## Paper Abstract

Machine Learning (ML) algorithms can suffer from biases, which may result in less favourable or unjust treatment across different demographic groups (e.g. females or ethnic minorities). Addressing these biases is crucial for ensuring fairness, yet it becomes complex when we need to achieve a balance between different fairness metrics. In this study, we conduct extensive and systematic experiments to explore the effectiveness of combining existing bias mitigation approaches to address multiple fairness metrics. We evaluate 12 bias mitigation approaches. For some of these approaches we include different configurations, as they can target different fairness metrics, resulting in a total of 26 different approaches. We then systematically combine these approaches, aiming to target multiple fairness objectives, resulting in more than 350 unique combinations. Our findings reveal that these combinations can help to improve fairness and can facilitate the simultaneous achievement of multiple fairness metrics. These findings carry significant implications for domains where meeting multiple fairness criteria is crucial, considering that most ML bias mitigation approaches typically focus on a single fairness metric, which may not fully address the multi-faceted nature of fairness in real-world contexts.

## Contents

This repository includes the following resources:

- **Paper (Future Update)**: Once the paper is reviewed, we plan to upload the camera-ready paper titled "Combining Machine Learning Bias Mitigation Approaches to Satisfy Multiple Fairness Constraints" which explores the proposed approach, methodology, experiments, and results.

- **Results**: The supplementary results discussed in the paper are included for reference.

- **Code**: Once the paper is reviewed, we plan to make public the code used for our experiments. The code is provided along with instructions to replicate the experiments and combine bias mitigation approaches in any tabular data.

## Repository structure

- **data**: Folder containing the data used for the experiments. Note that the _folktables_ data has to be downloaded using the code provided.
- **scripts**: Folder containing the scripts (and the Python file to generate the scripts) to replicate the experiments. The scripts execute all the possible bias mitigation approach combinations (the execution is done in parallel in batches of 5).
- **results**: Folder containing the experiments results.
  - **combination**: Contains JSON files with the results of the bias mitigation approaches combination. It also contains some graphs plotting the results in a matrix to facilitate their analysis.
  - **comparative**: Contains JSON files with the results of the individual bias mitigation approaches.
- ***apply_metrics.py***: Python file that contains the functions to compute the fairness metrics.
- ***fairness_comparative.py***: Python file that contains the functions to compute the bias mitigation approaches.
- ***fairness_utils.py***: Python file that contains the util functions.
- ***hybrid_fairness.py***: Python file that contains the functions to combine bias mitigation approaches.
- ***machine_learning.py***: Python file that contains the functions to apply machine learning algorithms.
- ***main.py***: Main Python file to execute a specific bias mitigation approach or combination of approaches.
- ***preprocessing.py***: Python file that contains the functions to process the data.
- ***requirements.txt***: File containing the requirements to execute our experiments.
- ***statistical_measures.py***: Python file that contains the functions to compute the statistical tests needed for some bias mitigation approaches.


## Example use

To apply a single bias mitigation approach:

```
python3 main.py comparative german gender DisparateImpactRemover --iterations 10
```

To combine two bias mitigation approaches:
```
python3 main.py hybrid german gender DisparateImpactRemover --mitigation_method2 ThresholdOptimizer_EqOdds --iterations 10
```

You can also apply all the experiments by executing the scripts in the **script** folder:
```
./german_experiments_script.sh  
```
