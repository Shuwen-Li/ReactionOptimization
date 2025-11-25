# ReactionOptimization
This is a repository for paper "Staged Diversity-Constrained Machine Learning for High-Dimensional Reaction Condition Optimization".

# Abstract
Optimizing reaction conditions in high-dimensional chemical spaces remains a central challenge in modern synthesis. In this context, we developed and evaluated a staged diversity-constrained machine learning framework that efficiently balances exploration and exploitation during condition optimization. At each stage, a within-batch diversity constraint promotes broad chemical coverage, while the constraint is progressively relaxed to focus on promising subspaces. Systematic evaluation across large-scale palladium-catalyzed C–C and C–N coupling datasets revealed that the number of stages, rather than the exploration portion, was the dominant factor governing optimization efficiency. A comparison with Bayesian optimization methods shows a dimension-dependent performance trend. Here, the staged diversity-constrained strategy was shown to be more advantageous in higher-dimensional reaction spaces, whereas Bayesian optimization performed better in lower-dimensional settings. Moreover, we developed a user-friendly software tool making the herein developed framework readily accessible for experimental chemists. Our strategy was further applied to challenging ruthenium-catalyzed meta-C–H functionalization involving 11,880 possible conditions, only 44 experiments were required to identify the optimal setup (91% yield). This work provides a validated and practical framework for accelerating high-dimensional reaction condition optimization, bridging data-driven modeling with experimental synthesis.

![workflow.jpg](https://github.com/Shuwen-Li/ReactionOptimization/blob/main/Figure/workflow.jpg)
# Packages requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below.
```
python = 3.8.0
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
The folder Bayesian demonstrates how to optimize reactions using Bayesian optimization.  
# Contact with us
Email: shuwen_li@zju.edu.cn
