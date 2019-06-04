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