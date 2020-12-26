import numpy as np
import pandas as pd
from plotnine import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import seaborn as sns
import pylab as pl
import os

class assignment3(object):
    def __init__(self):
        os.system('cls')
        print:"What do you want to do?"
        option = int(input("1. Outliers\n2.Mean Squared Error\n3. Local Maxima\n4. Autocorrelation\n5. Correlation & Linear Regression\n6. Classification\n7. Clustering\n"))
        if option == 1:
            self.solution1()
        elif option == 2:
            self.solution2()
        elif option == 3:
            self.solution3()
        elif option == 4:
            self.solution4()
        elif option == 5:
            self.solution5()
        elif option == 6:
            self.solutionClassification()
        elif option == 7:
            self.solutionCluster()
        else:
            print("Please select from options given.")
        self.__init__()
    
    def solution1(self):
        df = pd.read_csv('Effects-of-COVID-19-on-trade-1-February-21-October-2020-provisional.csv').dropna()
        df['nani'] = df['Year'].astype("category")
        vis = ggplot(df, aes(x='nani', y='Value')) + ggtitle('Covid') + theme(panel_spacing=0.5,figure_size=(10,5)) + \
            geom_boxplot()
        print(vis)
        os.system('pause')

    def solution2(self):
        df = pd.read_csv('Effects-of-COVID-19-on-trade-1-February-21-October-2020-provisional.csv')
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        regr = linear_model.LinearRegression()
        train_x = np.asanyarray(train[['Year']])
        train_y = np.asanyarray(train[['Value']])
        regr.fit (train_x, train_y)
        test_x = np.asanyarray(test[['Year']])
        test_y = np.asanyarray(test[['Value']])
        test_y_hat = regr.predict(test_x)
        print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
        print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
        print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
        os.system('pause')

    def solution3(self):
        df=pd.read_csv('cars.csv',sep=';')
        print(df.head())
        myColumns = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration']
        p=df[myColumns].max()
        q=df[myColumns].min()
        print("The local maximas are:")
        print(p)
        print("The local minimas are:")
        print(q)
        os.system('pause')

    def solution4(self):
        data = pd.read_csv('P1-US-Cities-Population.csv',encoding='latin1').dropna()
        x_values = data[['2010_Census']]
        y_values = data[['2015_estimate']]
        model = LinearRegression()
        model.fit(x_values, y_values)
        plt.scatter(x_values, y_values)
        plt.plot(x_values, model.predict(x_values))
        plt.show()
        os.system('pause')

    def solution5(self):
        df = pd.read_csv('cars.csv',sep='`;')
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        regr = linear_model.LinearRegression()
        train_x = np.asanyarray(train[['Model']])
        train_y = np.asanyarray(train[['Displacement']])
        regr.fit (train_x, train_y)
        print ('Coefficients: ', regr.coef_)
        print ('Intercept: ',regr.intercept_)
        os.system('pause')

    def solutionCluster(self):
        data=pd.read_csv('Effects-of-COVID-19-on-trade-1-February-21-October-2020-provisional.csv')
        f1=data['Value'].values
        f2=data['Cumulative'].values
        X=np.array(list(zip(f1,f2)))
        k=4
        kmeans= KMeans(n_clusters=k)
        kmeans= kmeans.fit(X)
        labels= kmeans.predict(X)
        centroids=kmeans.cluster_centers_
        colors = ['r','g','b','y','c','m','o','w']
        fig2 = plt.figure()
        kx=fig2.add_subplot(111)
        for i in range(k):
            points = np.array([X[j] for j in range(len(X))if labels[j]==i])
            kx.scatter(points[:,0],points[:,1],s=20,cmap='rainbow')
        kx.scatter(centroids[:,0],centroids[:,1],marker="*",s=200,c='#050505')
        print(centroids)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Number of clusters={}'.format(k))
        plt.show()
        os.system('pause')

    def solutionClassification(self):
        data = pd.read_csv('P1-US-Cities-Population.csv',encoding='latin1')
        print(data.head())
        print(data.shape)
        print(data['City'].unique())
        plt.show()
        data.drop('2015_rank', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                                title='Box Plot for each input variable')
        plt.show()
        data.drop('2015_rank' ,axis=1).hist(bins=30, figsize=(9,9))
        pl.suptitle("Histogram for each numeric input variable")
        plt.show()
        os.system('pause')

assignment3()
