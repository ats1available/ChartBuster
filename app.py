import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet,LinearRegression,ridge_regression,Lasso,ridge,Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from category_encoders import BinaryEncoder
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    features=np.array(features)
    features=features.reshape(1,6)
    features=pd.DataFrame(data=features,columns=['Name','Genre','Comments','Likes','Popularity','Followers'])
    df=pd.read_csv('data.csv')
    cv={'Comments':int,'Likes':int,'Popularity':int,'Followers':int}
    df=df.astype(cv)
    features=features.astype(cv)
    #x=df[df['Views']==0].index
    
    df.drop(index=df[df['Views']<df['Likes']].index,axis=1,inplace=True)
    df.drop(index=df[df['Views']<df['Comments']].index,axis=1,inplace=True)
    df.drop(index=df[df['Views']<df['Popularity']].index,axis=1,inplace=True)
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    (df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
    df=df[~((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
    
    df=df.drop(columns=['Unique_ID','Country','Song_Name','Timestamp','index'])
    
    y=df['Views']
    df=df.drop(columns=['Views'])
    
    be=BinaryEncoder()
    df=be.fit_transform(df)
    f=be.transform(features)
    
    X=df.iloc[:,:]
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=0)
    
    rg1=XGBRegressor()
    rg1.fit(X_train,y_train)
    ypred=rg1.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    
    rg2=AdaBoostRegressor()
    rg2.fit(X_train,y_train)
    ypred=rg2.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    
    rg3=ExtraTreesRegressor()
    rg3.fit(X_train,y_train)
    ypred=rg3.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    
    rg4=GradientBoostingRegressor(n_estimators=300,learning_rate=0.1)
    # para={'n_estimators':[250,300],'learning_rate':[1,0.1,0.01]}
    # grid=GridSearchCV(estimator=rg8,param_grid=para,verbose=1,cv=10,n_jobs=-1)
    rg4.fit(X_train,y_train)
    ypred=rg4.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    
    rg5=RandomForestRegressor(random_state=0,n_estimators=20,max_depth=15)
    # para={'n_estimators':[5,10,30,20],'max_depth':[5,8,20,17]}
    # grid=GridSearchCV(estimator=rg9,param_grid=para,cv=10,verbose=1,n_jobs=-1)
    rg5.fit(X_train,y_train)
    ypred=rg5.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    
    rg6=StackingRegressor([rg3,rg4,rg1,rg5],meta_regressor=rg5)
    rg6.fit(X_train,y_train)
    ypred=rg6.predict(X_test)
    #sqrt(mean_squared_error(y_test,ypred))
    f=f.iloc[:,:]
    y_pred=rg6.predict(f)
    
    y_pred=y_pred.astype(int)

    return render_template('index.html', prediction_text='Numberbof Views is {}'.format(y_pred))



if __name__ == "__main__":
    app.run(debug=True)