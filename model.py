import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Ridge
# from sklearn.cluster import KMeans
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster import MeanShift

import sys
import time

class model:
    def __init__(self, data, percent_drop=3, size_columns=5, train_test_size=0.7):
        self.data = data
        self.size_rows = int((len(self.data)) * train_test_size)  # porcentagem entre teste e treino
        self.percent_drop = (percent_drop / 10)
        self.size_columns = size_columns 

    def build_model(self):

        df_train, df_test, real_situation, next_day = self.build_data()

        x_train, x_test, y_train, y_test = self.define_target(df_train)
        clf, accuracy = self.train(x_train, x_test, y_train, y_test)
        print(accuracy)

        for i in [df_test, real_situation]:

            x_train, x_test, y_train, y_test = self.define_target(i)
            accuracy = self.test(x_train, x_test, y_train, y_test, clf)
            print(accuracy)

        self.predict_next_day(next_day, clf)
        

    def build_data(self):

        df_train = self.build_structure(0, (self.size_rows - self.size_columns), 1)
        df_test =  self.build_structure(self.size_rows, (len(self.data) - self.size_columns), 1)
        real_situation = self.build_structure(self.size_rows, (len(self.data) - self.size_columns), self.size_columns)
        next_day = self.build_structure((len(self.data) - self.size_columns + 1), (len(self.data) + 1))

        return [df_train, df_test, real_situation, next_day]

    def build_structure(self, origin, size, sep=False):

        df = []
        target = []

        if (sep == False):
            sep = 200
            
        for i in range(origin, size, sep):
            df.append(self.data.iloc[i:i + self.size_columns - 1, :].values.ravel()) 

            if(sep <= self.size_columns):
                target.append(self.data['Close'][i + self.size_columns - 1:i + self.size_columns].values)
                
        df = pd.DataFrame(df)

        if(sep <= self.size_columns):
            df['target'] = pd.DataFrame(target)

        df = df.sample(frac=1).reset_index().iloc[:, 1:]

        return df


    def define_target(self, df):

        y = df.iloc[:, -1:]

        x = df.iloc[:, :-1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=337)
        
        return [x_train, x_test, y_train, y_test]


    def train(self, x_train, x_test, y_train, y_test):

        # polynomialFeatures = PolynomialFeatures(degree=2)
        # X_train = polynomialFeatures.fit_transform(x_train)
        # X_test = polynomialFeatures.fit_transform(x_test)

        clf = GradientBoostingRegressor(loss='ls', max_depth=3, max_features='auto') 
        # clf = build_model(x_train) 
        # clf = LinearRegression() 
        # clf = RandomForestRegressor()
        # clf = KNeighborsRegressor()
        # clf = Ridge()
        # clf = KMeans()
        # clf = MiniBatchKMeans()
        # clf = MeanShift()

        # clf.fit(pd.DataFrame(X_train).iloc[:, 1:], y_train.values.ravel()) 
        # ypred = clf.predict(pd.DataFrame(X_test).iloc[:, 1:])

        clf.fit(x_train, y_train.values.ravel()) 
        ypred = clf.predict(x_test)

        percent = self.accuracy(ypred, x_test, y_test)

        return clf, percent

    def test(self, x_train, x_test, y_train, y_test, clf):
        
        ypred = clf.predict(x_test)

        percent = self.accuracy(ypred, x_test, y_test)

        return percent

    def predict_next_day(self, data, clf):

        # polynomialFeatures = PolynomialFeatures(degree=2)
        # data = polynomialFeatures.fit_transform(data)
        # predict = clf.predict(pd.DataFrame(data).iloc[:, 1:])

        predict = clf.predict(data)

        data = np.array(data)[0]
        vector = []

        # VALOR
        print("%.3f" % predict[0], end='\n')

        # ALTA OU BAIXA
        # for i, j in zip(np.arange(len(data)), data):
        #     if( i in [2, 6, 10, 14]):
        #         print("%.3f  >>  " % j, end="")

        # if (data[14] > predict[0]):
        #     print('DOWN')
        # else:
        #     print('UP')

    def accuracy(self, ypred, x_test, y_test):
        
        y_test.columns = ['Next_day']
        y_test['Predict_next_day'] = ypred

        y_test['Last_day'] = x_test.iloc[:, -2:-1]
        y_test['Nxt_day'] = [1 if (i[0] - j) >= 0 else 0 for i, j in zip(y_test.iloc[:, 0:1].values, y_test['Last_day'])]
        y_test['Predict_nxt_day'] = [1 if i >= 0 else 0 for i in (y_test['Predict_next_day'] - y_test['Last_day']) ]

        y_test['gain'] = y_test['Nxt_day'] == y_test['Predict_nxt_day']
        percent = len(y_test[y_test['gain'] == True]) * 100 / len(y_test['gain'])

        # print(y_test)

        return percent