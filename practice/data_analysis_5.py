
# coding: utf-8

# In[1]:


from __future__ import division
from numpy.random import randn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
np.random.seed(12345)
from pandas import DataFrame,Series
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4)
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pwd')


# In[2]:


plt.plot([1,2,1,1,4,4,7,6,48])


# In[3]:


plt.plot([1,2,3,4],[4,3,2,1])


# In[4]:


x = [1,2,3,4]
y = [5,4,3,2]
plt.figure()
plt.subplot(231)
plt.plot(x,y)

plt.subplot(232)
plt.bar(x, y)

plt.subplot(233)
plt.barh(x, y)

plt.subplot(234)
plt.bar(x, y)
y1 = [7,8,5,3]
plt.bar(x, y1, bottom=y, color = 'r')

plt.subplot(235)
plt.boxplot(x)

plt.subplot(236)
plt.scatter(x,y)

plt.show()


# In[5]:


#figure对象
fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax1.hist(randn(100), bins=20, color='k', alpha=0.3) # bins为间距，alpha 为线性厚度
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
ax3.plot(randn(50).cumsum(), 'k--')
fig.show()


# In[6]:


plt.figure()
plt.plot(x,y,"r--")
# plt.plot(x,y,linetype = "--",color = "r")


# In[7]:


plt.plot(randn(30).cumsum(), 'go--')

plt.plot(randn(30).cumsum(),color='k',linestyle='dashed',marker='o')


# In[8]:


data = randn(30).cumsum()
plt.close("all")
plt.plot(data,"k--",drawstyle="steps-post",label="steps-post")
plt.legend(loc = "best")


# In[9]:


plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())
ax.set_xticks([0,250,500,750,1000])
ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')
fig.show()


# In[10]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(100).cumsum(),"k--",label="one")
ax.plot(randn(100).cumsum(),"k.",label="two")
ax.plot(randn(100).cumsum(),"k",label="three")
fig.legend(loc="best")


# In[11]:


#注释以及在subplot上绘图
from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
inputfile = open('F:/电影/练数成金/第7周/data/spx.csv')
data = pd.read_csv(inputfile, index_col=0, parse_dates=True)
spx = data['SPX']
ax.plot(data,"g--")
fig.savefig("F:/fig.png",dpi=1080,bbox_inchs="tight")


# In[12]:


plt.close("all")
s = Series(randn(10).cumsum(),index=np.arange(0,100,10))
s.plot()


# In[13]:


df = DataFrame(np.random.randn(10, 4).cumsum(0),
               columns=['A', 'B', 'C', 'D'],
               index=np.arange(0, 100, 10))
df.plot()


# In[14]:


#柱形图
fig, axes = plt.subplots(2, 1)
data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)


# In[15]:


df = DataFrame(np.random.rand(6, 4),
               index=['one', 'two', 'three', 'four', 'five', 'six'],
               columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot(kind='bar')


# In[16]:


df.plot(kind="bar",stacked=True,alpha=0.5)


# In[17]:


inputfile2 = open('F:/电影/练数成金/第7周/data/tips.csv')
tips = pd.read_csv(inputfile2)
party_counts = pd.crosstab(tips.day, tips['size'])
party_counts


# In[18]:


party_counts = party_counts.ix[:, 2:5]
party_pcts = party_counts.div(party_counts.sum(1).astype(float),axis = 0)
party_pcts


# In[19]:


party_pcts.plot(kind="bar",stacked=True)


# In[20]:


tips["tip-pct"] = tips['tip']/tips['total_bill']
tips['tip-pct'].hist(bins=50)


# In[21]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(tips["tip-pct"],color="b",bins = 50)


# In[22]:


comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')


# In[23]:


#散点图
inputfile3 = open('F:/电影/练数成金/第7周/data/macrodata.csv')
macro = pd.read_csv(inputfile3)
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
pd.scatter_matrix(trans_data,diagonal="kde",color="k",alpha="0.5")


# In[24]:


#误差条形图
x = np.arange(0, 10, 1)

y = np.log(x)

xe = 0.1 * np.abs(np.random.randn(len(y)))

plt.bar(x, y, yerr=xe, width=0.4, align='center', ecolor='r', color='cyan',
                                                    label='experiment #1');

plt.xlabel('# measurement')
plt.ylabel('Measured values')
plt.title('Measurements')
plt.legend(loc='upper left')

plt.show()


# In[25]:


#饼图
plt.figure(1, figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.8, 0.8])

labels = 'Spring', 'Summer', 'Autumn', 'Winter'
values = [15, 16, 16, 28]
explode =[0.1, 0.1, 0.1, 0.1]

plt.pie(values, explode=explode, labels=labels,
    autopct='%1.1f%%', startangle=67)

plt.title('Rainy days by season')

plt.show()


# In[26]:


#等高线图
import matplotlib as mpl

def process_signals(x, y):
    return (1 - (x ** 2 + y ** 2)) * np.exp(-y ** 3 / 3)

x = np.arange(-1.5, 1.5, 0.1)
y = np.arange(-1.5, 1.5, 0.1)

X, Y = np.meshgrid(x, y)

Z = process_signals(X, Y)

N = np.arange(-1, 1.5, 0.3)

CS = plt.contour(Z, N, linewidths=2, cmap=mpl.cm.jet)
plt.clabel(CS, inline=True, fmt='%1.1f', fontsize=10)
plt.colorbar(CS)

plt.title('My function: $z=(1-x^2+y^2) e^{-(y^3)/3}$')
plt.show()


# In[27]:


# 3D 图像
import matplotlib.dates as mdate
from mpl_toolkits.mplot3d import Axes3D


# In[28]:


mpl.rcParams['font.size'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for z in [2011, 2012, 2013, 2014]:
    xs = range(1,13)
    ys = 1000 * np.random.rand(12)
    
    color = plt.cm.Set2(np.random.choice(range(plt.cm.Set2.N)))
    ax.bar(xs, ys, zs=z, zdir='y', color=color, alpha=0.8)

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))


ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_zlabel('Sales Net [usd]')

plt.show()


# In[29]:


#3d直方图
mpl.rcParams['font.size'] = 10

samples = 25

x = np.random.normal(5, 1, samples)
y = np.random.normal(3, .5, samples)

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

hist, xedges, yedges = np.histogram2d(x, y, bins=10)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(xedges[:-1]+.25, yedges[:-1]+.25)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)

dx = .1 * np.ones_like(zpos)
dy = dx.copy()

dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.4)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

ax2 = fig.add_subplot(212)
ax2.scatter(x, y)
ax2.set_xlabel('X Axis')
ax2.set_ylabel('Y Axis')

plt.show()


# In[31]:


# 对macrodata.csv数据集
# 1. 画出realgdp列的直方图
# 2. 画出realgdp列与realcons列的散点图，初步判断两个变量之间的关系
inputfile3 = open('F:/电影/练数成金/第7周/data/macrodata.csv')
macro = pd.read_csv(inputfile3)
data1 = macro["realgdp"]
data2 = macro[["realgdp","realcons"]]
pd.scatter_matrix(data2,diagonal="kde",color="k",alpha="0.5")


# In[33]:


data1.hist(bins = 100)


# In[37]:


# 对tips数据集
# 3. 画出不同sex与day的交叉表的柱形图
# 4. 画出size的饼图
inputfile2 = open('F:/电影/练数成金/第7周/data/tips.csv')
tips = pd.read_csv(inputfile2)
party_sex = pd.crosstab(tips.day, tips.sex)  
party_pcts=party_sex.div(party_sex.sum(1).astype(float), axis=0)  
party_pcts.plot(kind='bar', stacked=True)  


# In[1]:


#饼图
plt.figure(1, figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.8, 0.8])

labels='1','2','3','4','5','6'  
values=[]  
for i in range(6):  
 values.append(sum(tips['size']==i+1))  
explode =[0.1, 0.1, 0.1, 0.1,0.1,0.1] 

plt.pie(values, explode=explode, labels=labels,
    autopct='%1.1f%%', startangle=67)

plt.title('Rainy days by season')

plt.show()

