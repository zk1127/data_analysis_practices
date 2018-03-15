
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


inputfile = u"F:/电影/练数成金/第13周/sales.xlsx"
data = pd.read_excel(inputfile)
print(data.head())
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)
#拆分训练数据与测试数据 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[3]:


#训练KNN分类器 
clf = KNeighborsClassifier(n_neighbors = 4,algorithm='kd_tree')
clf.fit(x_train, y_train)


# In[4]:


#测试结果
answer = clf.predict(x_test)
print(x_test)
print(y_test)
print(answer)
print(np.mean(answer == y_test))


# In[7]:


precision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
print(classification_report(y_test,answer,target_names=['高','低']))


# In[9]:


clf = BernoulliNB()
clf.fit(x_train,y_train)
answer = clf.predict(x_test)
print(x_test)
print(y_test)
print(answer)
print(np.mean(answer == y_test))


# In[10]:


from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')
dtc.fit(x_train,y_train)
#测试结果
answer = dtc.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean( answer == y_test))
print(classification_report(y_test, answer, target_names = ['低', '高']))


# In[11]:


from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train,y_train)
#测试结果
answer = dtc.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean( answer == y_test))
print(classification_report(y_test, answer, target_names = ['低', '高']))


# In[12]:


print(classification_report(y_test, answer, target_names = ['低', '高']))

