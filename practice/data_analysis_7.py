
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


inputfile = open('F:/电影/练数成金/第10周/Advertising.csv')
data = pd.read_csv(inputfile,index_col=0)
data.head()


# In[3]:


import matplotlib


# In[4]:


#计算相关系数矩阵
data.corr()


# In[10]:


#构建X、Y数据集
X = data[['TV', 'radio', 'newspaper']]

y = data['sales']

##直接根据系数矩阵公式计算
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


#求解回归方程系数
X2=X
X2['intercept']=[1]*200
standRegres(X2,y)


# In[12]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)
print(linreg.intercept_) # 截距
print(linreg.coef_) # 系数


# In[14]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
#结果
print(linreg.intercept_)
print(linreg.coef_)
print(zip(['TV','Radio','Newspaper'], linreg.coef_))

#预测
y_pred = linreg.predict(X_test)
y_pred


# In[15]:


#误差评估
from sklearn import metrics

# calculate MAE using scikit-learn
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))


# calculate MSE using scikit-learn
print ("MSE:",metrics.mean_squared_error(y_test,y_pred))


# calculate RMSE using scikit-learn
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

