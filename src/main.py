import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import LinearSVC


# test.csv and train.csv location
test_data_src = "/home/hyder/ds/porto/test.csv"
train_data_src = "/home/hyder/ds/porto/train.csv"


# Load the test and train csv
#test_data = pd.read_csv(test_data_src)
train_data = pd.read_csv(train_data_src)

# To make things easier, let's label our test and train data before merging them
#test_data['Type'] = 'Test'

# merge the test and train data
#full_data = pd.concat([train_data,test_data],axis=0)

train_data = train_data.head(200000)
full_data = train_data
test_data = train_data.tail(200000)
train_data['Type'] = 'Train'
test_data['Type'] = 'Test'

# since the dataset is very large, let's start by working on small part of it [1000 entries from each data set]
#full_data = pd.concat([full_data.head(100000),full_data.tail(100000)],axis=0)



# recognise the ID field, and target [target is 0 or 1 i.e whether the driver claims insurance or not]
id_col = ['id']
target_col = ['target']
# total number of columns
num_cols= list(set(list(full_data.columns))-set(id_col)-set(target_col))

# this is our own very recently created column
other_col = ['Type']

# divide the data in test and train here
#train=full_data[full_data['Type']=='Train']
#test=full_data[full_data['Type']=='Test']

train = train_data
test = test_data


# feature is all other columns except target, id and Type column that we created
features=list(set(list(full_data.columns))-set(id_col)-set(target_col)-set(other_col))

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

acc=(y_true == predict).sum()/((y_true == predict).sum()+(y_true != predict).sum())
print(acc)
