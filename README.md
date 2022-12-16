# Semi-supervised-Learning Tutorial

## 목차

### 0. Overview of Semi-Supervised Learning
___
### 1. Hybrid/Holistic Methods
#### 1-1. MixMatch
#### 1-2. FixMatch
#### 1-3. FlexMatch
___
### 2. Experimental Analysis
### 3. Conclusion
___
## 0. Overview of Ensemble
![image](https://user-images.githubusercontent.com/115224653/204128272-c65bf7d7-a25e-491b-a06a-cfde1837ac0f.png)
### - What is "Ensemble?"
Ensemble is a French word for unity and harmony. It is mainly used in music to mean concerto on various instruments. 

A large number of small instrumental sounds are harmonized to create a more magnificent and beautiful sound. 

Of course, you shouldn't, but one tiny mistake can be buried in another sound.    

Ensemble in machine learning is similar. Several weak learners gather to form a stronger strong learner through voting. 

Since there are many models, even if the prediction is misaligned in one model, it is somewhat corrected. That is, a more generalized model is completed.   

The two goals of the ensemble are as follows.

### 1. How do we ensure diversity?
### 2. How do you combine different results from different models?
___
## Dataset

We use 7 datasets for Classification (Banking, Breast, Diabetes, Glass, PersonalLoan, Stellar, Winequality)   

Banking dataset : <https://www.kaggle.com/datasets/rashmiranu/banking-dataset-classification>     
Breast dataset : <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>   
Diabetes dataset : <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>   
Glass datset : <https://www.kaggle.com/datasets/uciml/glass>      
PersonalLoan dataset : <https://www.kaggle.com/datasets/teertha/personal-loan-modeling>   
Steallr dataset : <https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17>   
WineQuality dataset : <https://archive.ics.uci.edu/ml/datasets/wine+quality>   

In all methods, data is used in the following form.
``` C
import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='4_Ensemble')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='Banking.csv',
                        choices = ['Banking.csv', 'Breast.csv', 'Diabetes.csv', 'Glass.csv', 'PersonalLoan.csv', 'Steallr.csv', 'WineQuality.csv'])            
                        
data = pd.read_csv(args.data_path + args.data_type)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
```
___

# Bagging

## What is "Bagging"?   
   

![Bagging](https://www.simplilearn.com/ice9/free_resources_article_thumb/Bagging.PNG)

Bagging is one of the enamble methods that increases the performance of the model.  

If you look at the above picture, you can see that the same data with the same color is duplicated in bootstrap.   

 Like this, Bagging creates multiple subset datasets while allowing duplicate extraction from the original dataset, which is called "Bootstrap", and that is why Bagging is also called Bootstrap aggregating.  

 Bagging is characterized by learning models in parallel using Bootstrap, and with this, Bagging is used to deal with bias-variance trade-offs and reduces the variance of a prediction model.

### Bagging is effective when using base learners with high model complexity.   

Bagging avoids overfitting of data and is used for both regression and classification models, specifically for decision tree algorithms.   

This tutorial covers the case of using Decision Tree and Random Forest as base learners by applying Bagging.   

In the analysis part, an ablation study is conducted on the performance difference between when bagging is applied and when not applied.
___
   
## 1. Decision Tree (DT)

<p align="center"><img src="https://regenerativetoday.com/wp-content/uploads/2022/04/dt.png" width="650" height="400"></p> 

Decision tree analyzes the data and represents a pattern that exists between them as a combination of predictable rules and is called a decision tree because it looks like a 'tree'.

The above example is a binary classification problem that determines yes=1 if the working conditions for the new job are satisfied and no=0 if not satisfied.

As shown in the picture above, the initial point is called the root node and the corresponding number of data decreases as the branch continues.

Decision trees are known for their high predictive performance compared to computational complexity.

In addition, it has the strength of explanatory power in units of variables, but decision trees are likely to work well only on specific data because the decision boundary is perpendicular to the data axis.

The model that emerged to overcome this problem is Random Forest, a technique that improves predictive performance by combining the results by creating multiple decision trees for the same data.
___
## Python Code
``` py
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def Decision_Tree(args):

    result_df = pd.DataFrame(columns = ['Bagging', 'Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for bagging in args.bagging:
        for depth in args.max_depth_list:
            for seed in args.seed_list:

                data = pd.read_csv(args.data_path + args.data_type)

                X_data = data.iloc[:, :-1]
                y_data = data.iloc[:, -1]

                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_data)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

                if bagging == True:
                    model = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = depth),
                                                                n_estimators=args.n_estimators,
                                                                random_state = seed)
                else:
                    model = DecisionTreeClassifier(max_depth=depth)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_pred, y_test)
                f1score = f1_score(y_pred, y_test, average='weighted')
                
                print('Decision Tree max_depth =', depth)
                print('Accuracy :', accuracy, 'F1-Score :', f1score)

                result = {'Bagging' : bagging,
                            'Seed' : seed,
                            'Depth' : depth,
                            'Accuracy' : accuracy,
                            'F1-Score' : f1score}
                
                result = pd.DataFrame([result])

                result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df
```
Unlike the previous tutorials, this tutorial conducted repeated experiments. 

A total of five random seed values were allocated and repeated. 

In the middle, the args.bagging part is for comparing the performance when bagging is applied and when not applied. 

In the Decision Tree part, the experimental construction is how much performance has improved when bagging is applied. 

This part will be examined in the analysis part later.

___

## 2. Random Forest
<p align="center"><img src="https://i0.wp.com/thaddeus-segura.com/wp-content/uploads/2020/09/rfvsdt.png?fit=933%2C278&ssl=1" width="900" height="300"></p>

Random Forest is an enamble model that improves predictive power by reducing the correlation of individual trees by taking advantage of existing bagging and adding a process of randomly selecting variables.   

Let's take a closer look at the above definition below.

### 1. Random Forest is an ensemble model using Bagging.
Random forest basically uses Bagging. Therefore, Random Forest will also take over the effect of lowering dispersion while maintaining the bias, which is the advantage of Bagging.

### 2. Random Forest improves prediction by reducing the correlation of individual trees through the process of randomly selecting variables.
Random Forest uses the Bootstrap sample dataset to create several individual trees. In Breiman et al.'s 'Random Forest' paper, it is proved that a smaller correlation between individual trees results in a smaller generalization error of the random forest. In other words, reducing the correlation of individual trees means that the predictive power of the random forest is improved.

### It is important to randomly select candidates for variables to separate individual trees!
___

#### Python Code
``` py
def Random_Forest(args):

    result_df = pd.DataFrame(columns = ['Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for depth in args.max_depth_list:
        for seed in args.seed_list:

            data = pd.read_csv(args.data_path + args.data_type)

            X_data = data.iloc[:, :-1]
            y_data = data.iloc[:, -1]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_data)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

            model = RandomForestClassifier(max_depth = depth,
                                            n_estimators=args.n_estimators,
                                            random_state = seed)
                                            
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_pred, y_test)
            f1score = f1_score(y_pred, y_test, average='weighted')
            
            print('Decision Tree max_depth =', depth)
            print('Accuracy :', accuracy, 'F1-Score :', f1score)

            result = {
                        'Seed' : seed,
                        'Depth' : depth,
                        'Accuracy' : accuracy,
                        'F1-Score' : f1score}
            
            result = pd.DataFrame([result])

            result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df
```
Random Forest part will cover the comparison of performance with the case of Decision Tree without Bagging and Decision Tree with Bagging, not the experiment of tuning the hyperparameter.
___
# Boosting

## What is "Boosting"?   
   

![Boosting](https://velog.velcdn.com/images/iguv/post/3bc28cf5-23f8-4c8b-b509-52c25382f564/image.png)

Boosting is an ensemble method that combines weak learners with poor performance to build a good performance model, and the sequentially generated weak learners compensate for the shortcomings of the previous step in each step.

The training method depends on the type of boosting process called the boosting algorithm. 

However, the algorithm trains the boosting model by following the following general steps:

1. The boosting algorithm assigns the same weight to each data sample. It supplies data to the first machine model, called the basic algorithm. The basic algorithm allows you to make predictions for each data sample.

2. The boosting algorithm evaluates model predictions and increases the weight of samples with more serious errors. It also assigns weights based on model performance. Models that produce excellent predictions have a significant impact on the final decision.

3. The algorithm moves the weighted data to the next decision tree.

4. The algorithm repeats steps 2 and 3 until the training error instance falls below a certain threshold.

### Boosting is effective when using base learners with low model complexity!   
___
## 3. Adaptive Boosting (AdaBoost)

<p align="center"><img src="https://cdn-images-1.medium.com/max/800/1*7TF0GggFTqetjxqU5cnuqA.jpeg" width="750" height="300"></p> 

The Adaboost algorithm is a classification-based machine learning model, a method of synthesizing one strong classifier that performs better by weight modification by building and combining a large number of weak classifiers with slightly lower predictive performance. 

The Adaboost model has the advantage of repeatedly modifying and combining weights through mistakes in weak classifiers, and not compromising predictive performance due to less overfitting of the learning data.

In other words, it is the principle of generating a final strong classifier by adding the product of the weight of the weak classifier and the value of the weak classifier.
___
#### Python Code
``` py
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

def AdaptiveBoosting(args):

    result_df = pd.DataFrame(columns = ['Boosting', 'Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for boosting in args.boosting:
        for depth in args.max_depth_list:
            for seed in args.seed_list:

                data = pd.read_csv(args.data_path + args.data_type)

                X_data = data.iloc[:, :-1]
                y_data = data.iloc[:, -1]

                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_data)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

                if boosting == True:
                    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = depth),
                                                                n_estimators=args.n_estimators,
                                                                random_state = seed)
                else:
                    model = DecisionTreeClassifier(max_depth=depth)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_pred, y_test)
                f1score = f1_score(y_pred, y_test, average='weighted')
                
                print('Decision Tree max_depth =', depth)
                print('Accuracy :', accuracy, 'F1-Score :', f1score)

                result = {'Boosting' : boosting,
                            'Seed' : seed,
                            'Depth' : depth,
                            'Accuracy' : accuracy,
                            'F1-Score' : f1score}
                
                result = pd.DataFrame([result])

                result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df
```
In the middle, the args.boosting part is for comparing the performance when boosting is applied and when not applied. Like the Decision Tree part, the results of the Decision Tree with Adaptive Boosting and the results of the Decision Tree without Adaptive Boosting will be compared in the analysis part.
___
## 4. Gradient Boosting Machine (GBM)

<p align="center"><img src="https://www.akira.ai/hubfs/Imported_Blog_Media/akira-ai-gradient-boosting-ml-technique.png" width="750" height="300"></p> 

Gradient Boosting Machines (GBM) is a way to understand the concept of Boosting as an optimization method called Gradient Descent.

As I explained before, boosting is a method of adding and sequential learning multiple trees and synthesizing the results.

GBM, like AdaBoost, is an algorithm of the Boosting family, so it complements the residual of the previous learner by creating a weak learner sequentially, but the two have different methods of complementing the residual.

AdaBoost addresses previously misclassified data by giving more weight to well-classified learners.

By comparison, GBM is a method of updating predictions by fitting weak learners to the residuals themselves and adding the predicted residuals to the previous predictions.
___
#### Python Code
``` py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

def GradientBoosting(args):

    result_df = pd.DataFrame(columns = ['Seed', 'n_estimators', 'Accuracy', 'F1-Score'])


    for n_estimators in args.gbm_estimators:
        for seed in args.seed_list:

            data = pd.read_csv(args.data_path + args.data_type)

            X_data = data.iloc[:, :-1]
            y_data = data.iloc[:, -1]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_data)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

            model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                    random_state = seed)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_pred, y_test)
            f1score = f1_score(y_pred, y_test, average='weighted')
            
            print('Decision Tree n_estimators =', n_estimators)
            print('Accuracy :', accuracy, 'F1-Score :', f1score)

            result = {
                        'Seed' : seed,
                        'n_estimators' : n_estimators,
                        'Accuracy' : accuracy,
                        'F1-Score' : f1score}
            
            result = pd.DataFrame([result])

            result_df = pd.concat([result_df, result], ignore_index = True)

    return result_df
```
In the experimental part of Gradient Boosting Machines, we adjust the n_estimators hyperparameter and compare the resulting performance changes while adjusting the model complexity. 

This part will be examined in the analysis part.
___
## 5. eXtra Gradient Boost (XGBoost)
![image](https://user-images.githubusercontent.com/115224653/204450586-d5c53de5-4b3a-4fe7-adeb-ebd63e0c1b43.png)

Gradient Boost is a representative algorithm implemented using the Boosting technique.

The library that implements this algorithm to support parallel learning is eXtra Gradient Boost (XGBost).

It supports both Regression and Classification problems, and is a popular algorithm with good performance and resource efficiency.

Because it learns through parallel processing, the classification speed is faster than that of general GBM.

In addition, in the case of standard GBM, there is no overfitting regulation function, but XGBoost itself has strong durability as an overfitting regulation function.

It has an Early Stopping function, offers a variety of options, and is easy to customize.

___


### Python Code
``` py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

def ExtremeGradientBoosting(args):

    result_df = pd.DataFrame(columns = ['Seed', 'n_estimators', 'Accuracy', 'F1-Score'])

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    if args.data_type == 'WineQuality.csv':
        y_data -= 3

    elif args.data_type == 'Glass.csv':
        y_data -= 1
        
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = args.seed)

    model = XGBClassifier()

    xgb_param_grid = {
        'n_estimators' : args.gbm_estimators,
        'learning_rate' : args.xgb_lr,
        'max_depth' : args.xgb_depth
    }

    xgb_grid = GridSearchCV(model, param_grid = xgb_param_grid, scoring='accuracy', n_jobs = -1, verbose = 1)
    xgb_grid.fit(X_train, y_train)

    print('Best Accuracy : {0:.4f}'.format(xgb_grid.best_score_))
    print('Best Parameters :', xgb_grid.best_params_)

    result_df = pd.DataFrame(xgb_grid.cv_results_)
    result_df.sort_values(by=['rank_test_score'], inplace=True)

    result_df[['params', 'mean_test_score', 'rank_test_score']].head(5)

    return result_df        
```
In the experimental part of XGBoost, grid search for hyperparameters n_estimators, learning_rate, and max_depth was conducted.

Let's compare the best performance when each dataset has a hyperparameter.


___

## Analysis


## [Experiment 1.] Decision Tree - DT Performance Comparison by Bagging Application

In the Python code section of the decision tree, it was possible to set whether to apply bagging through the arg.bagging argument.

It is intended to understand the effect of model complexity and bagging on performance by comparing the performance according to the pre-set max_depth value with the application of bagging.

As for the performance evaluation metric, accuracy and F1 score were used as in the previous tutorial.

First of all, the table below is a performance table of the decision tree without bagging. 

### All experiments are the results of five repeated experiments by changing the seed value.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8345 (±0.008)** |        0.7881 (±0.005)|         0.7820 (±0.003)|        0.7824 (±0.004) |        0.7830 (±0.008)|
|          2 | Breast                 |    0.9368 (±0.022) |        0.9404 (±0.010)|         **0.9456 (±0.012)**|        0.9404 (±0.019) |        0.9404 (±0.030)|
|          3 | Diabetes                 |    **0.7013 (±0.025)**|        0.6857 (±0.040) |        0.6922 (±0.044) |        0.6896 (±0.036) |        0.6935 (±0.037) |
|          4 | Glass                 |   **0.6837 (±0.066)** |        0.6698 (±0.045) |        0.6744 (±0.033) |        0.6514 (±0.057) |        0.6514 (±0.004) |
|          5 | PersonalLoan                 |    0.9822 (±0.002)|        0.9814 (±0.001) |        0.9820 (±0.002) |        **0.9824 (±0.003)** |        0.9814 (±0.002) | 
|          6 | Stellar                 |    **0.9676 (±0.002)** |        0.9614 (±0.002) |        0.9599 (±0.002) |        0.9609 (±0.002) |        0.9608 (±0.002) | 
|          7 | WineQuality                 |    0.6069 (±0.027) |        0.6150 (±0.032) |        **0.6175 (±0.026)** |        0.6138 (±0.026) |        0.6125 (±0.025)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8602 (±0.003)** |        0.7814 (±0.003)|         0.7721 (±0.003)|        0.7729 (±0.003) |        0.7734 (±0.003)|
|          2 | Breast                 |    0.9368 (±0.022)|        0.9403 (±0.065)|         **0.9456 (±0.065)**|        0.9403 (±0.065) |        0.9404 (±0.065)|
|          3 | Diabetes                 |    **0.7001 (±0.026)**|        0.6820 (±0.043) |        0.6894 (±0.046) |        0.6864 (±0.039) |        0.6912 (±0.039) |
|          4 | Glass                 |   0.6801 (±0.037) |        0.6723 (±0.027) |        0.6776 (±0.027) |        **0.7034 (±0.027)** |        0.6978 (±0.027) |
|          5 | PersonalLoan                 |    0.9822 (±0.003)|        0.9814 (±0.003) |        0.9820 (±0.003) |        **0.9824 (±0.003)** |        0.9814 (±0.003) | 
|          6 | Stellar                 |    **0.9677 (±0.002)** |        0.9613 (±0.002) |        0.9598 (±0.002) |        0.9608 (±0.002) |        0.9607 (±0.002) | 
|          7 | WineQuality                 |    0.6127 (±0.026) |        0.6155 (±0.030) |        **0.6181 (±0.026)** |        0.6159 (±0.023) |        0.6147 (±0.022)  |

The table below shows the performance table of the decision tree to which bagging is applied.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8607 (±0.003)** |        0.8494 (±0.003)|         0.8476 (±0.003)|        0.8475 (±0.003) |        0.8475 (±0.003)|
|          2 | Breast                 |    **0.9647 (±0.007)** |        0.9631 (±0.065)|         0.9632 (±0.065)|        0.9632 (±0.065) |        0.9632 (±0.065)|
|          3 | Diabetes                 |    **0.7688 (±0.017)**|        0.7649 (±0.020) |        0.7649 (±0.020) |        0.7649 (±0.020) |        0.7649 (±0.020) |
|          4 | Glass                 |   **0.7395 (±0.037)** |        0.7256 (±0.027) |        0.7256 (±0.027) |        0.7256 (±0.027) |        0.7256 (±0.027) |
|          5 | PersonalLoan                 |    **0.9884 (±0.003)**|        0.9880 (±0.003) |        0.9880 (±0.003) |        0.9880 (±0.003) |        0.9880 (±0.003) | 
|          6 | Stellar                 |    0.9761 (±0.002) |        **0.9767 (±0.002)** |        0.9766 (±0.002) |        0.9766 (±0.002) |        0.9766 (±0.002) | 
|          7 | WineQuality                 |    0.6725 (±0.021) |        0.6763 (±0.022) |        **0.6788 (±0.025)** |        0.6787 (±0.025) |        0.6787 (±0.025)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.9119 (±0.003)** |        0.8840 (±0.005)|         0.8801 (±0.005)|        0.8801 (±0.005) |        0.8801 (±0.005)|
|          2 | Breast                 |    **0.9632 (±0.006)** |        0.9632 (±0.065)|         0.9632 (±0.065)|        0.9632 (±0.065) |        0.9632 (±0.065)|
|          3 | Diabetes                 |    **0.7727 (±0.020)**|        0.7692 (±0.022) |        0.7692 (±0.022) |        0.7692 (±0.022) |        0.7692 (±0.022) |
|          4 | Glass                 |   **0.7491 (±0.037)** |        0.7359 (±0.028) |        0.7359 (±0.028) |        0.7359 (±0.028) |        0.7359 (±0.028) |
|          5 | PersonalLoan                 |    **0.9885 (±0.003)**|        0.9882 (±0.003) |        0.9882 (±0.003) |        0.9882 (±0.003) |        0.9882 (±0.003) | 
|          6 | Stellar                 |    0.9763 (±0.002) |        **0.9768 (±0.002)** |        0.9767 (±0.001) |        0.9767 (±0.001) |        0.9767 (±0.001) | 
|          7 | WineQuality                 |    0.6902 (±0.022) |        0.6939 (±0.020) |        **0.6969 (±0.023)** |        0.6969 (±0.023) |        0.6969 (±0.023)  |

Analyzing the experimental results can be summarized as follows.

#### 1. Compared to when bagging was not applied, performance was improved on all datasets when applied.
#### 2. When the model complexity is high, the performance of bagging is generally good. However, the performance was rather good when the max_depth value related to the model complexity was lower than when it was high.
#### 3. Setting the max_depth hyperparameter value seems to have an important effect on performance.
#### 4. The deviation between repeated experiments was low.
___

## [Experiment 2.] Random Forest - Performance Comparison with Depth values

Random forest uses bagging and random variable extraction techniques. 

And there is a difference in performance depending on the model complexity, and we observe the performance comparison by changing the depth value associated with the model complexity.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8671 (±0.003)** |        0.8608 (±0.003)|         0.8603 (±0.003)|        0.8601 (±0.003) |        0.8601 (±0.003)|
|          2 | Breast                 |    **0.9614 (±0.007)** |        0.9614 (±0.007)|         0.9614 (±0.007)|        0.9614 (±0.007) |        0.9614 (±0.007)|
|          3 | Diabetes                 |    **0.7584 (±0.011)**|        0.7571 (±0.014) |        0.7545 (±0.016) |        0.7545 (±0.016) |        0.7545 (±0.016) |
|          4 | Glass                 |   0.7628 (±0.023) |        **0.7674** (±0.042) |        0.7674 (±0.042) |        0.7674 (±0.042) |        0.7674 (±0.042) |
|          5 | PersonalLoan                 |   0.9886 (±0.003)|        0.9886 (±0.003) |        **0.9888 (±0.003)** |        0.9888 (±0.003) |        0.9888 (±0.003) | 
|          6 | Stellar                 |    0.9750 (±0.002) |        0.9752 (±0.001) |        **0.9756 (±0.001)** |        0.9756 (±0.001) |        0.9756 (±0.001) | 
|          7 | WineQuality                 |    0.6788 (±0.019) |        0.6913 (±0.025) |        **0.6900 (±0.025)** |        0.6900 (±0.025) |        0.6900 (±0.025)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.9272 (±0.002)** |        0.9100 (±0.003)|         0.9086 (±0.004)|        0.9086 (±0.004) |        0.9086 (±0.004)|
|          2 | Breast                 |    **0.9615 (±0.007)** |        0.9615 (±0.008)|         0.9615 (±0.008)|        0.9615 (±0.008) |        0.9615 (±0.008)|
|          3 | Diabetes                 |    **0.7645 (±0.015)**|        0.7640 (±0.017) |        0.7614 (±0.019) |        0.7614 (±0.019) |        0.7614 (±0.019) |
|          4 | Glass                 |   **0.7763 (±0.018)** |        0.7812 (±0.040) |        0.7812 (±0.040) |        0.7812 (±0.040) |        0.7812 (±0.040) |
|          5 | PersonalLoan                 |    **0.9889 (±0.003)**|        0.9888 (±0.002) |        0.9890 (±0.002) |        0.9890 (±0.002) |        0.9890 (±0.002) | 
|          6 | Stellar                 |    0.9751 (±0.002) |        **0.9753 (±0.001)** |        0.9757 (±0.001) |        0.9757 (±0.001) |        0.9757 (±0.001) | 
|          7 | WineQuality                 |    0.7011 (±0.017) |        0.7101 (±0.024) |        **0.7089 (±0.022)** |        0.7089 (±0.022) |        0.7089 (±0.022)  |

Analyzing the experimental results can be summarized as follows.

#### 1. Random Forest did not always perform better than the Decision Tree with Bagging.
#### 2. As in Decision Tree, an increase in max_depth associated with model complexity did not improve performance.
#### 3. Similarly, it was observed that setting the max_depth hyperparameter value suitable for the dataset is important.

___
## [Experiment 3.] AdaBoost - DT Performance Comparison by Adaptive Boosting Application


As confirmed in the python code part, AdaBoost also utilized the decision tree as the base learner and the case where Adaptive Boosting is applied and not.

Like experiment 1, this time, we would like to compare performance changes depending on the application of Boosting (AdaBoost).

The table below to which Adaptive Boosting is not applied is the same as the table of results when Bagging is not applied in experiment 1.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8345 (±0.008)** |        0.7881 (±0.005)|         0.7820 (±0.003)|        0.7824 (±0.004) |        0.7830 (±0.008)|
|          2 | Breast                 |    0.9368 (±0.022) |        0.9404 (±0.010)|         **0.9456 (±0.012)**|        0.9404 (±0.019) |        0.9404 (±0.030)|
|          3 | Diabetes                 |    **0.7013 (±0.025)**|        0.6857 (±0.040) |        0.6922 (±0.044) |        0.6896 (±0.036) |        0.6935 (±0.037) |
|          4 | Glass                 |   **0.6837 (±0.066)** |        0.6698 (±0.045) |        0.6744 (±0.033) |        0.6514 (±0.057) |        0.6514 (±0.004) |
|          5 | PersonalLoan                 |    0.9822 (±0.002)|        0.9814 (±0.001) |        0.9820 (±0.002) |        **0.9824 (±0.003)** |        0.9814 (±0.002) | 
|          6 | Stellar                 |    **0.9676 (±0.002)** |        0.9614 (±0.002) |        0.9599 (±0.002) |        0.9609 (±0.002) |        0.9608 (±0.002) | 
|          7 | WineQuality                 |    0.6069 (±0.027) |        0.6150 (±0.032) |        **0.6175 (±0.026)** |        0.6138 (±0.026) |        0.6125 (±0.025)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8602 (±0.003)** |        0.7814 (±0.003)|         0.7721 (±0.003)|        0.7729 (±0.003) |        0.7734 (±0.003)|
|          2 | Breast                 |    0.9368 (±0.022)|        0.9403 (±0.065)|         **0.9456 (±0.065)**|        0.9403 (±0.065) |        0.9404 (±0.065)|
|          3 | Diabetes                 |    **0.7001 (±0.026)**|        0.6820 (±0.043) |        0.6894 (±0.046) |        0.6864 (±0.039) |        0.6912 (±0.039) |
|          4 | Glass                 |   0.6801 (±0.037) |        0.6723 (±0.027) |        0.6776 (±0.027) |        **0.7034 (±0.027)** |        0.6978 (±0.027) |
|          5 | PersonalLoan                 |    0.9822 (±0.003)|        0.9814 (±0.003) |        0.9820 (±0.003) |        **0.9824 (±0.003)** |        0.9814 (±0.003) | 
|          6 | Stellar                 |    **0.9677 (±0.002)** |        0.9613 (±0.002) |        0.9598 (±0.002) |        0.9608 (±0.002) |        0.9607 (±0.002) | 
|          7 | WineQuality                 |    0.6127 (±0.026) |        0.6155 (±0.030) |        **0.6181 (±0.026)** |        0.6159 (±0.023) |        0.6147 (±0.022)  |

And the table below is a table of performance results when Adaptive Boosting is applied.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8524 (±0.003)** |        0.8248 (±0.028)|         0.8087 (±0.033)|        0.8087 (±0.033) |        0.8087 (±0.033)|
|          2 | Breast                 |    **0.9439 (±0.022)** |        0.9439 (±0.022)|         0.9439 (±0.022)|        0.9439 (±0.022) |        0.9439 (±0.022)|
|          3 | Diabetes                 |    **0.7638 (±0.011)**|        0.6858 (±0.039) |        0.6858 (±0.039) |        0.6858 (±0.039) |        0.6858 (±0.039) |
|          4 | Glass                 |   0.6686 (±0.027) |        0.6768 (±0.061) |        0.6768 (±0.061) |        0.6768 (±0.061) |        **0.6768 (±0.061)** |
|          5 | PersonalLoan                 |    **0.9882 (±0.002)**|        0.9828 (±0.001) |        0.9828 (±0.001) |        0.9828 (±0.001) |        0.9828 (±0.001) | 
|          6 | Stellar                 |    **0.9745 (±0.002)** |        0.9681 (±0.006) |        0.9606 (±0.002) |        0.9606 (±0.002) |        0.9606 (±0.002) | 
|          7 | WineQuality                 |    **0.6738 (±0.028)** |        0.5975 (±0.017) |        0.6075 (±0.020) |        0.6075 (±0.020) |        0.6075 (±0.020)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8943 (±0.004)** |        0.8453 (±0.050)|         0.8194 (±0.061)|        0.8185 (±0.059) |        0.8181 (±0.059)|
|          2 | Breast                 |    **0.9437 (±0.022)**|        0.9437 (±0.022)|         0.9437 (±0.022)|        0.9437 (±0.022) |        0.9437 (±0.022)|
|          3 | Diabetes                 |    **0.7638 (±0.011)**|        0.6858 (±0.039) |        0.6858 (±0.039) |        0.6858 (±0.039) |        0.6858 (±0.039) |
|          4 | Glass                 |   0.6686 (±0.027) |        **0.6768 (±0.061)** |        0.6768 (±0.061) |        0.6768 (±0.061) |        0.6768 (±0.061) |
|          5 | PersonalLoan                 |    **0.9885 (±0.002)**|        0.9828 (±0.001) |        0.9828 (±0.001) |        0.9828 (±0.001) |        0.9828 (±0.001) | 
|          6 | Stellar                 |    **0.9747 (±0.002)** |        0.9682 (±0.006) |        0.9605 (±0.002) |        0.9605 (±0.002) |        0.9605 (±0.002) | 
|          7 | WineQuality                 |    **0.6920 (±0.029)** |        0.5974 (±0.017) |        0.6087 (±0.018)|        0.6087 (±0.018) |        0.6087 (±0.018)  |

Analyzing the experimental results can be summarized as follows.

#### 1. Compared to when Adaptive Boosting was not applied, performance was improved when Adaptive Boosting was applied.
#### 2. The Boosting family is effective when the model complexity is low, and unlike the results of Bagging's experiments, Adaptive Boosting performed best when the depth=10 has the relatively lowest model complexity.
#### 3. Glass dataset did not show the effect of boosting.
___
## [Experiment 4.] Gradient Boosting Machine (GBM) - Performance comparison based on changes in n_estimators values

GBM is also a type of Boosting and is effective for algorithms with low model complexity. 

One of GBM's arguments, n_estimators, is a hyperparameter that determines how many times boosting is applied. 

Personally, I think the model complexity will be high when n_estimators are high, so I observe how the performance changes while changing the n_estimators value.

Since the default value of the n_estimators hyperparameter is 100, we will increase it from 100.

|  Accuracy  | Dataset              |  n_estimators = 100 |   n_estimators = 200 |   n_estimators = 300 |  n_estimators = 400 |   n_estimators = 500 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8647 (±0.004)** |        0.8586 (±0.006)|         0.8549 (±0.006)|        0.8509 (±0.006) |        0.8471 (±0.006)|
|          2 | Breast                 |    0.9649 (±0.011) |        **0.9667 (±0.013)**|         0.9667 (±0.013)|        0.9649 (±0.015) |        0.9649 (±0.015)|
|          3 | Diabetes                 |    **0.7506 (±0.011)**|        0.7312 (±0.013) |        0.7364 (±0.013) |        0.7364 (±0.018) |        0.7286 (±0.023) |
|          4 | Glass                 |   0.7535 (±0.048) |        **0.7628 (±0.040)** |        0.7628 (±0.045) |        0.7628 (±0.040) |        0.7628 (±0.040) |
|          5 | PersonalLoan                 |    0.9886 (±0.003)|        0.9890 (±0.002) |        **0.9898 (±0.002)** |        0.9890 (±0.002) |        0.9888 (±0.002) | 
|          6 | Stellar                 |    0.9743 (±0.002)|        0.9751 (±0.001) |        0.9757 (±0.001) |        0.9759 (±0.001) |        **0.9762 (±0.001)** | 
|          7 | WineQuality                 |    0.6550 (±0.022) |        0.6700 (±0.013) |        **0.6763 (±0.011)** |        0.6750 (±0.009) |        0.6731 (±0.016)  |

|  F1-Score  | Dataset              |  n_estimators = 100 |   n_estimators = 200 |   n_estimators = 300 |  n_estimators = 400 |   n_estimators = 500 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.9199 (±0.005)** |        0.9080 (±0.005)|         0.8990 (±0.005)|        0.8909 (±0.006) |        0.8835 (±0.006)|
|          2 | Breast                 |    0.9649 (±0.011) |        **0.9668 (±0.013)**|         0.9667 (±0.013)|        0.9651 (±0.015) |        0.9651 (±0.015)|
|          3 | Diabetes                 |    **0.7566 (±0.013)**|        0.7375 (±0.014) |        0.7419 (±0.016) |        0.7415 (±0.021) |        0.7335 (±0.024) |
|          4 | Glass                 |   0.7611 (±0.038) |        **0.7738 (±0.036)** |        0.7727 (±0.041) |        0.7719 (±0.035) |        0.7719 (±0.035) |
|          5 | PersonalLoan                 |    0.9887 (±0.003)|        0.9891 (±0.002) |        **0.9899 (±0.002)** |        0.9891 (±0.002) |        0.9889 (±0.002) | 
|          6 | Stellar                 |    0.9745 (±0.002)|        0.9752 (±0.001) |        0.9759 (±0.001) |        0.9760 (±0.001) |        **0.9763 (±0.001)** | 
|          7 | WineQuality                 |    0.6685 (±0.022) |        0.6826 (±0.028) |        **0.6893 (±0.016)** |        0.6877 (±0.015) |        0.6860 (±0.017)  |

Analyzing the experimental results can be summarized as follows.

#### 1. The n_estimators value showed the highest performance near 300. That is, it is insignificant whether n_estimators directly affect the model complexity.
#### 2. Certain datasets showed better performance than in the case of Adaptive Boosting.

___
## [Experiment 5.] XGBoost - Comparison performance with GridSearch

In the case of XGBoost, there are three hyperparameters: n_estimator, learning_rate, and max_depth. 

Therefore, in this experiment, various hyperparameter combinations are formed and the best combination is found through gridsearch. 

And in the next experimental part, based on all experimental results to date, we summarize the algorithms with the best performance for each dataset.

| Index | Dataset      | n_estimators | learning_rate | max_depth | Accuracy |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 1     | Banking      | 200          | 0.01          | 4         | 0.8668   |
| 2     | Breast       | 200          | 0.05          | 4         | 0.9670   |
| 3     | Diabetes     | 300          | 0.01          | 4         | 0.7557   |
| 4     | Glass        | 100          | 0.05          | 4         | 0.7366   |
| 5     | PersonalLoan | 100          | 0.1           | 4         | 0.9858   |
| 6     | Stellar      | 500          | 0.1           | 6         | 0.9776   |
| 7     | WineQuality  | 100          | 0.2           | 10        | 0.6599   |

## [Experiment 5-1.] Best Algorithm for each Dataset (Accuracy)

| Index | Dataset      | Algorithm         | n_estimators  | learning_rate | max_depth   | Accuracy |
|:-------:|:--------------:|:-------------------:|:---------------:|:---------------:|:-------------:|:----------:|
| 1     | Banking      | Random Forest     | 100 (default) | X             | 10          | 0.8671   |
| 2     | Breast       | XGBoost           | 200           | 0.05          | 4           | 0.9670   |
| 3     | Diabetes     | Bagging(DT)       | 100 (default) | X             | 10          | 0.7688   |
| 4     | Glass        | Random Forest     | 100 (default) | X             | 20          | 0.7674   |
| 5     | PersonalLoan | Gradient Boosting | 300           | 0.1 (default) | 3 (default) | 0.9898   |
| 6     | Stellar      | XGBoost           | 500           | 0.1           | 6           | 0.9776   |
| 7     | WineQuality  | Random Forest     | 100 (default) | X             | 30          | 0.6900   |

___
## Conclusion

### 1) Appropriate hyperparameter changes were needed to indicate changes in model performance according to model complexity.
### 2) Random Forest, famous for its good performance, performed best on three of the seven datasets.
### 3) The Boosting family showed high performance when the model complexity was low, but the Bagging family did not prove high performance when the model complexity was high.
### 4) Just as the best algorithms are different for each dataset, I felt that it was worth using various algorithms when used in practice.
___

### Reference

- Business Analytics, Korea university (IME-654) https://www.youtube.com/watch?v=vlkbVgdPXc4&t=1588s
- https://www.simplilearn.com/tutorials/machine-learning-tutorial/bagging-in-machine-learning
- https://regenerativetoday.com/simple-explanation-on-how-decision-tree-algorithm-makes-decisions/The-anomaly-detection-and-the-classification-learning-schemas_fig1_282309055
- https://www.researchgate.net/publication/345327934/figure/fig3/AS:1022810793209856@1620868504478
- https://www.akira.ai/hubfs/Imported_Blog_Media/
- https://cdn-images-1.medium.com
- https://i0.wp.com/thaddeus-segura.com/

