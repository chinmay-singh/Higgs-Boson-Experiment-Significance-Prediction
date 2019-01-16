# HiggsML_CUHK
## The Scoring metric has been kept as the ams as given in the question
## The model has been defined in classifier.py
### The model used was ADABoost because 
1) it supports weighted examples in Scikit learn library.
2) It is adequately suited for binary classification problems.
3) It is good at telling the model to focus on the wronged examples because the iven metric penalises false positives more than false neatives.
### The model parameters were found by Distributed Asynchronous Hyper-parameter Optimization by defining a searchspace using hyperopt library
>n_estimators: 20. Hyperopt value. I tried 15, 20, 25, etc. 20 gave optimal results
>learning_rate: 0.75. I tried 1.0, 0.9, 0.8, etc. (.6 and .7 gave similar results)

## ExtraTreesClassifier:
I picked ExtraTreesClassifier because since there are many examples,
it benefits from subsampling. I tried GradientBoostingClassifier is too slow
and RandomForestClassifier does not take advantage of the subsampling.
Subsampling is good when training data is abundant.

## Parameters:
n_estimators: 400.Found using Hyperopt. More trees to an extend is usually
 good.
max_features: 30.Found using Hyperopt. Using more features is usually a good
idea, noting that using fewer features increases the randomness,
which makes the model good. If you use too few features for each
tree, there will not be enough predictive power in each tree.
max_depth: 12. Found using Hyperopt. This can overfit really easy.
min_samples_leaf: 100. Found using Hyperopt. I tried tuning this up because
 we have many data instances to learn from. It reflected a positive affect on the result.
min_samples_split: 100. Found using Hyperopt. I tried tuning this up because
 we have many data instances to learn from. It reflected a positive affect on the result.
 
 ## Preprocessing 
 I did not get to try many new preprocessing, I can explore new imputation methods and outlier removal in the due course of time
 I tried makin new features using deep feature synthesis as well as made new feature sets of derived parameter but it did not help.
 I simply imputed most frequent entry for the missing data.
 I did scaling of the data (took inverse log of the positive columns) and used standard scaler on the entire data.
 
 ## Training and Prediction
 I decided to use a thresholdin cut of 83 percent on the predictions to predict the positives bearing the scoring metric in mind.
 
 Way forward
>Exploiting the physics effectively.
>Major improvements on the preprocessing part can be made
