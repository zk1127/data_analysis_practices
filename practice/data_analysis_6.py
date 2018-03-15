
# coding: utf-8

# In[2]:


from __future__ import division
from numpy.random import randn
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)


# In[3]:


df1 = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})
df1


# In[4]:


group1 = df1["data1"].groupby(df1["key1"])
group1


# In[6]:


group1.mean()


# In[10]:


means1 = df1["data1"].groupby([df1["key1"],df1["key2"]]).mean()
means1


# In[12]:


means1.unstack()


# In[13]:


df1.groupby(['key1', 'key2']).size()


# In[15]:


# ### 对分组进行迭代
for name, group in df1.groupby('key1'):
    print(name)
    print(group)


# In[17]:


grouped = df1.groupby(df1.dtypes, axis=1)
dict(list(grouped))


# In[18]:


# ### 通过字典或series进行分组
people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan # Add a few NA values
people


# In[19]:


mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}
by_column = people.groupby(mapping,axis = 1)
by_column.sum()


# In[20]:


# ### 通过函数进行分组
people.groupby(len).sum()

key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len,key_list]).min()


# In[21]:


# ### 通过索引进行分组
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
hier_df


# In[23]:


hier_df.groupby(level = "cty",axis = 1).count()


# In[26]:


# ### 面向列的多函数应用
inputfile1 = open('F:/电影/练数成金/第8周/data/tips.csv')
tips = pd.read_csv(inputfile1)

tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]


# In[30]:


group2 = tips.groupby(["sex","smoker","size","time"])
group2.count()


# In[31]:


k1_means = df1.groupby('key1').mean().add_prefix('mean_')
k1_means


# In[33]:


key = ['one', 'two', 'one', 'two', 'one']
def demean(arr):
    return arr - arr.mean()
demeaned = people.groupby(key).transform(demean)
demeaned


# In[34]:


# ### apply方法
def top(df, n=5, column='tip_pct'):
    return df.sort_index(by=column)[-n:]
top(tips, n=6)


# In[35]:


tips.groupby('smoker').apply(top)


# In[36]:


tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')


# In[37]:


inputfile2 = open('F:/电影/练数成金/第8周/data/stock_px.csv')
close_px = pd.read_csv(inputfile2, parse_dates=True, index_col=0)
close_px.info()


# In[38]:


close_px[-4:]


# In[41]:


rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)
by_year.apply(spx_corr)
# 苹果公司和微软的年度相关系数
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


# In[42]:


tips.pivot_table(index=['sex', 'smoker'])


# In[43]:


tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'],
                 columns='smoker')


# In[44]:


tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day',
                 aggfunc=len, margins=True)


# In[48]:


# ### 交叉表
from io import StringIO
data = """Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')

data


# In[49]:


pd.crosstab(data.Gender, data.Handedness, margins=True)


# In[50]:


# 交叉表
pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)


# In[52]:


inputfile3 = open('F:/电影/练数成金/第8周/data/P00000001-ALL.csv')
fec = pd.read_csv(inputfile3)
print(fec.info())
fec.ix[123456]


# In[53]:


unique_cands = fec.cand_nm.unique()
unique_cands


# In[54]:


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
fec.cand_nm[123456:123461]

fec.cand_nm[123456:123461].map(parties)
fec['party'] = fec.cand_nm.map(parties)
fec['party'].value_counts()


# In[56]:


fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels


# In[57]:


grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)


# In[59]:


grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
totals[:10]

percent = totals.div(totals.sum(1), axis=0)
percent[:10]


# In[65]:


# 读入tips.csv 数据集
# 1. 统计不同time的tip均值，方差
print((tips.groupby(["time"]).mean())["tip"])
(tips.groupby(["time"]).var())["tip"]


# In[67]:


# 2. 将total_bill和tip根据不同的sex进行标准化
tips.groupby(["sex"]).std()


# In[70]:


# 3. 计算吸烟者和非吸烟者的小费比例值 均值的差值

#添加小费比例值列
tips['tip_pct'] = tips['tip'] / tips['total_bill']
grouped1 = tips.groupby(['smoker'])[['tip_pct']]
#求吸烟者和非吸烟者的小费比例值均值
result1 = grouped1.mean()
#定义差值函数
def peak_to_peak(arr):
    return arr.max() - arr.min()
print(result1)
#计算吸烟者和非吸烟者的小费比例值均值的差值
result1.agg([peak_to_peak])


# In[78]:


# 4. 对sex与size聚合，统计不同分组的小费比例的标准差、均值，将该标准差与均值添加到原数据中

tips.groupby(["sex","size"]).agg("std")["tip_pct"]
# tips["mean_ss"] = tips.groupby(["sex","size"]).agg("mean")
# print(tips["std_ss"],"\n",tips["mean_ss"])
# tips[:6]


# In[87]:


# 5. 对time和size聚合，画出total_bill 的饼图
size = tips.groupby(["sex","size"])["total_bill"].agg("count")


# In[90]:


#饼图
plt.figure(1, figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.8, 0.8])

labels='f1','f2','f3','f4','f5','f6','m1','m2','m3','m4','m5','m6'  
values=[]  
for s in size:  
 values.append(s)  
explode =[0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 

plt.pie(values, explode=explode, labels=labels,
    autopct='%1.1f%%', startangle=67)

plt.title('Rainy days by season')

plt.show()

