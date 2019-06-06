# Reports

## 4.6.19
### Initial pathology classifier
We have built a classifier for distinguishing pathologies,
Based on McPAS-TCR pathology-associated database, and the TCR autoencoder.

**The model process** - We use the trained autoencoder to encode each TCR. the encoded vector 
is fed into MLP to predict one pathology within multiple classes. 
We can control the number of classes in the code, however we take only k-top most frequent pathologies.

**Current performance:**
On 10 classes,

dataset| accuracy (100 epochs) | accuracy (200 epochs)
--- | --- | ---
train (80%) | 0.58 | 0.63
test (20%) | 0.49 | 0.48

:grimacing:

Since the data is imbalanced, we plan to use over-sampling methods for the minority classes,
such as SMOTE.

## 5.6.19
### SMOTE over-sampling
We used SMOTE for creating synthetic examples of the minority classes.
First, we use the TCR-autoencoder in order to extract encoding vector for all examples.
Next, we use SMOTE to equal the number of samples in all classes.
(SMOTE is used only for the train set).
The expanded trained data is fed to a simple MLP model with one hidden layer.

**Current performance:**
On 10 classes,

dataset | accuracy (15 epochs)
--- | ---
train (80%) | 0.54
test (20%) | 0.30

:frowning_face:	

It is worse than before because it is a simple MLP model without autoencoder extra training.
In addition the model is very slow because we did not added batching yet.
It seems like the SMOTE caused overfitting.

Next, we will try building a model which will also train the autoencoder.
We also would like to use automatic hyperparameters tuning.

## 6.6.19
### AutoML
We used the auto-sklearn package to find the best model for the classification problem.
We run an AutoML process on the training data with and without SMOTE over-sampling.
The prediction is an output of an ensemble of models,
currently I did not figure out how to read what models were used.

**Current performance:**
On 10 classes,

dataset | sampling | accuracy
--- | --- | ---
train (80%) | with train SMOTE |0.97
test (20%) | with train SMOTE |0.43
train (80%) | without train SMOTE |0.83
test (20%) | without train SMOTE |0.51

:four_leaf_clover:


### K-Nearest Neighbors
In addition, in order to see if clustering algorithms are relevant for this data,
and in particular if SMOTE is a valid over-sampling method, we tried using simple KNN algorithm
(on the original encoding vectors extracted from the TCR autoencoder).

**Current performance:**
On 10 classes,

dataset | number of neighbors | accuracy
--- | --- | ---
train (80%) | 1 |0.98
test (20%) | 1 |0.46
train (80%) | 3 |0.66
test (20%) | 3 |0.49
train (80%) | 5 |0.61
test (20%) | 5 |0.49
train (80%) | 7 |0.58
test (20%) | 7 |0.50

:open_mouth:
