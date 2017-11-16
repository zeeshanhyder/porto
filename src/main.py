import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import LinearSVC


test_data_src = "/home/hyder/ds/porto/test.csv"
train_data_src = "/home/hyder/ds/porto/train.csv"

test_data = pd.read_csv(test_data_src)
train_data = pd.read_csv(train_data_src)

test_data['Type'] = 'Test'
train_data['Type'] = 'Train'

full_data = pd.concat([train_data,test_data],axis=0)

full_data = pd.concat([full_data.head(1000),full_data.tail(1000)],axis=0)


id_col = ['id']
target_col = ['target']
num_cols= list(set(list(full_data.columns))-set(id_col)-set(target_col))
other_col = ['Type']



train=full_data[full_data['Type']=='Train']
test=full_data[full_data['Type']=='Test']



features=list(set(list(full_data.columns))-set(id_col)-set(target_col)-set(other_col))

x_train = train[list(features)].values
y_train = train['target'].values


x_test = test[list(features)].values


clf = LinearSVC(random_state=0)
clf.fit(x_train, y_train)

print( clf.predict(x_test) )