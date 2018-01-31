import scipy.io
from sklearn.ensemble import RandomForestClassifier as RandomForests
import numpy as np
import os
from util import load_single_train_data




rf = RandomForests(500)

train_x, train_y, test_x, test_y = load_single_train_data('.cache\dba\data', 1)

rf.fit(train_x, train_y)
pr_test_y = rf.predict(test_x)

right = 0
for i, y in enumerate(test_y):
    if y == pr_test_y[i]:
        right += 1
print(right/len(test_y))

'''

'''