import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import LinearSVC
import csv


TEST_PORTION=.5
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


for var in num_cols:
    if full_data[var].isnull().any()==True:
        full_data[var+'_NA']=full_data[var].isnull()*1

full_data[num_cols] = full_data[num_cols].fillna(full_data[num_cols].mean(),inplace=True)



train=full_data[full_data['Type']=='Train']
test=full_data[full_data['Type']=='Test']



train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

features=list(set(list(full_data.columns))-set(id_col)-set(target_col)-set(other_col))

x_train = Train[list(features)].values
y_train = Train['target'].values

x_validate = Validate[list(features)].values
y_validate = Validate['target'].values

x_test = test[list(features)].values


clf = LinearSVC(random_state=0)
clf.fit(x_train, y_train)

print( clf.predict(x_test) )