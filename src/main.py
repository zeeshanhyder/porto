import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import LinearSVC


# dataset size to work on
SIZE = 200000
# test.csv and train.csv location
data_src = "/home/hyder/ds/porto/train.csv"


# Load the test and train csv
data = pd.read_csv(data_src)

# To make things easier, let's label our test and train data before merging them

# divide the data in train and test
train = data.head(SIZE)
test = data.tail(SIZE)
train['Type'] = 'Train'
test['Type'] = 'Test'


# recognise the ID field, and target [target is 0 or 1 i.e whether the driver claims insurance or not]
id_col = ['id']
target_col = ['target']
# total number of columns
num_cols= list(set(list(data.columns))-set(id_col)-set(target_col))
# this is our own very recently created column
other_col = ['Type']



# feature is all other columns except target, id and Type column that we created
features=list(set(list(data.columns))-set(id_col)-set(target_col)-set(other_col))

# X and y of our data to train on
x_train = train[list(features)].values
y_train = train['target'].values


# This is what we will use in predict to test our classifier against [test data]
x_test = test[list(features)].values
y_true = test['target'].values



# LinearSVC classifier [Later on we will use SGD]
clf = linear_model.SGDClassifier(n_jobs=3)
# fit
clf.fit(x_train, y_train)

# print the prediction if driver will claim insurance or not
predict = clf.predict(x_test)

# calculate the accuracy num_of_correct_predictions/total_predictions
acc=(y_true == predict).sum()/((y_true == predict).sum()+(y_true != predict).sum())

print("Accuracy for",SIZE," entries:")
print(acc)
