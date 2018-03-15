
# coding: utf-8

# In[1]:


from pandas import Series,DataFrame
import pandas as pd


# In[2]:


inputfile = 'F:/电影/练数成金/第11周/bankloan.xls'
data = pd.read_excel(inputfile)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()
data


# In[3]:


from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR # 随机逻辑回归模型
rlr = RLR()
rlr.fit(x,y)


# In[4]:


rlr.get_support()


# In[5]:


import numpy as np
print(u'通过随机逻辑回归模型筛选特征结束。')
array = np.array([False, False,  True,  True, False,  True,  True, False,False])
print(u'有效特征为：%s' % ','.join(data.columns[array]))
x = data[data.columns[array]].as_matrix() #筛选好特征


# In[6]:


lr = LR()
lr.fit(x,y)
print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s' % lr.score(x, y)) #给出模型的平均正确率，本例为81.4%


# In[7]:


import matplotlib.pyplot as plt
from sklearn import metrics
x=pd.DataFrame([1.5,2.8,4.5,7.5,10.5,13.5,15.1,16.5,19.5,22.5,24.5,26.5])
y=pd.DataFrame([7.0,5.5,4.6,3.6,2.9,2.7,2.5,2.4,2.2,2.1,1.9,1.8])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x,y)


# In[8]:


from sklearn.linear_model import LinearRegression as Linereg
linereg = Linereg()
linereg.fit(x,y)
print('Coefficients: \n', linereg.coef_)
y_pred = linereg.predict(x)


# In[9]:


print("MSE:",metrics.mean_squared_error(y,y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linereg.score(x, y))


# In[10]:


x2=pd.DataFrame(np.log(x[0]))

linreg = Linereg()
linreg.fit(x2,y)

# The coefficients
print('Coefficients: \n', linreg.coef_)

y_pred = linreg.predict(x2)
# The mean square error
print("MSE:",metrics.mean_squared_error(y,y_pred))
print('Variance score: %.2f' % linereg.score(x2, y))


# In[13]:


# 1. data1 是40名癌症病人的一些生存资料，其中，X1表示生活行动能力评分（1~100），X2表示病人的年龄，
# X3表示由诊断到直入研究时间（月）；X4表示肿瘤类型，X5把ISO两种疗法（“1”是常规，“0”是试验新疗法）；
# Y表示病人生存时间（“0”表示生存时间小于200天，“1”表示生存时间大于或等于200天）
# 试建立Y关于X1~X5的logistic回归模型
path = 'F:/电影/练数成金/第11周/data1.txt'
data = str(open(path))
data


# In[14]:


from io import StringIO
data = pd.read_table(StringIO(data), sep='\s+')
data


# In[22]:


x = data.iloc[1:,:6].as_matrix()
y = data.iloc[:,6].as_matrix()
y


# In[23]:


# 2. data2 是关于重伤病人的一些基本资料。自变量X是病人的住院天数，因变量Y是病人出院后长期恢复的预后指数，
# 指数数值越大表示预后结局越好。
# 尝试对数据拟合合适的线性或非线性模型
x=pd.DataFrame([2,5,7,10,14,19,26,31,34,38,45,52,53,60,65])
y = pd.DataFrame([54,50,45,3,35,25,20,16,18,13,8,11,8,4,6])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x,y)


# In[30]:


# 多项式
from sklearn.linear_model import LinearRegression
x1=x
x2=x**2
x1['x2']=x2
linereg = LinearRegression()
linereg.fit(x1,y)
print('Coefficients: \n', linreg.coef_)
y_pred = linereg.predict(x)
print("MSE:",metrics.mean_squared_error(y,y_pred))
print('Variance score: %.2f' % linreg.score(x, y))


# In[31]:


#对数
x2 = pd.DataFrame(np.log(x))
linereg = LinearRegression()
linereg.fit(x2,y)
print('Coefficients: \n', linreg.coef_)
y_pred = linereg.predict(x2)
print("MSE:",metrics.mean_squared_error(y,y_pred))
print('Variance score: %.2f' % linreg.score(x2, y))


# In[33]:


# 指数
y2=pd.DataFrame(np.log(y))

linreg = LinearRegression()
linreg.fit(pd.DataFrame(x[0]),y2)

# The coefficients
print('Coefficients: \n', linreg.coef_)

y_pred = linreg.predict(pd.DataFrame(x[0]))
# The mean square error
print("MSE:",metrics.mean_squared_error(y2,y_pred))
print('Variance score: %.2f' % linreg.score(x, y))


# In[29]:


#幂函数

linreg = LinearRegression()
linreg.fit(x2,y2)

# The coefficients
print('Coefficients: \n', linreg.coef_)

y_pred = linreg.predict(x2)
# The mean square error
print("MSE:",metrics.mean_squared_error(y2,y_pred))

