import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
from sklearn.svm import SVR
svr_lin = SVR(kernel='linear',C=1e3)
svr_poly = SVR(kernel='poly',C=1e3, degree=2)
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)

def load_data():
    work_data = pd.read_csv('train.csv')
    print "work dataset has {} data points with {} variables each.".format(*work_data.shape)
    # print housing_data.dtypes
    return work_data


def get_features(data):
    features = data.drop(['Vehicles'], axis = 1)
    features['DateTime'] = pd.to_datetime(features['DateTime'])
    # features['Junction'] = features['ID']
    features = features.set_index('DateTime')

    return features

def get_vehicles(data):
    vehicles = data['Vehicles']
    return vehicles


data = load_data()
features = get_features(data)
vehicles = get_vehicles(data)

print features.shape
print vehicles.shape

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#
X_train, X_test, y_train, y_test = train_test_split(features, vehicles, test_size=0.4,  random_state=0)
regressor.fit(X_train,y_train)
svr_rbf.fit(X_train,y_train)
svr_poly.fit(X_train,y_train)
pred0 = regressor.predict(X_test)
pred1 = svr_rbf.predict(X_test)
pred3 = svr_poly.predict(X_test)
print r2_score(y_test, pred0)
print r2_score(y_test, pred1)
print r2_score(y_test, pred3)