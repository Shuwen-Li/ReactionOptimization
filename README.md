# ReactionOptimization
This is a repository for paper "Integrating Orthogonal Design with Machine Learning for High-Dimensional Reaction Optimization".

# Abstract
Optimizing reaction conditions in high-dimensional chemical spaces remains a significant challenge in chemical synthesis. In this study, we present a novel strategy that integrates orthogonal experimental design with machine learning to efficiently optimize reaction conditions. Our approach balances exploration and exploitation by embedding orthogonal constraints, which guide the model towards diverse sampling, while progressively focusing on the most promising regions of the synthetic space. We systematically evaluated key modeling parameters, including the number of optimization stages and the proportion of the exploration space at each stage, and found that the multi-stage design with progressive constraint relaxation maximized optimization efficiency. Additionally, automated machine learning was employed for descriptor selection and algorithm tuning, further enhancing the optimization process. We applied our strategy to a Ru-catalyzed meta-C–H functionalization reaction, optimizing a space of 11,880 possible conditions. After only 44 experiments, the model identified optimal reaction conditions, achieving a 91% yield—substantially higher than the previously reported literature conditions. This study highlights the potential of integrating orthogonal design with machine learning to accelerate the discovery and optimization of synthetic methodologies, providing a robust framework for high-dimensional reaction optimization in molecular synthesis.

![workflow.jpg](https://github.com/Shuwen-Li/ReactionOptimization/blob/main/Figure/workflow.jpg)
# Packages requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below.
```
autosklearn = 0.15.0
matplotlib = 3.7.2
numpy = 1.24.4
pandas = 2.0.3
seaborn = 0.13.2
sklearn = 0.24.2
torch = 1.4.0
xgboost = 2.1.1
```

# Demo & Instructions for use
Notebook Examples.ipynb demonstrates how to use our work to reaction optimization.
Notebook results.ipynb demonstrates the reults of reaction optimization.
Notebook path.ipynb demonstrates the path of reaction optimization.
# Contact with us
Email: shuwen_li@zju.edu.cn
