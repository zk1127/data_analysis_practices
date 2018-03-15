
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100


# In[2]:


# 拉格朗日插值
inputfile = u"F:/电影/练数成金/第6周/data/catering_sale.xls" # 测试文件路径
outfile = u"F:/电影/练数成金/第6周/data/out.xls" # 输出文件路径
data = pd.read_excel(inputfile) #读入数据
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None #过滤异常值，将其变为空值


# In[3]:


print(data)


# In[4]:


#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
# def ployinterp_column(s,n,k = 5): # k为步长
#     y = s[list(range(n-k,n))+list(range(n+1,n+1+k))]
#     y = y[y.notnull()] # 剔除空值
#     return lagrange(y.index,list(y))(n)
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) # 最后的括号就是我们要插值的n


# In[5]:


# for i in data.columns:
#     for  j in range(len(data)):
#         if (data[i].isnull())[j]:
#             ployinterp_column(data[i],j)
# data.to_excel(outfile)
#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i].isnull())[j]: #如果为空即插值。
      data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outfile) #输出结果，写入文件
print(data)


# In[6]:


###dataframe合并
#1
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
print(df1)
print(df2)


# In[7]:


pd.merge(df1,df2)
pd.merge(df1,df2,on = "key")


# In[9]:


df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
pd.merge(df3,df4,left_on="lkey",right_on="rkey")


# In[11]:


# 外联接
print(pd.merge(df1,df2,how="outer"))
#左联接
pd.merge(df1,df2,on ="key" , how = "left")
# 默认内联接
pd.merge(df1,df2,how = "inner")


# In[12]:


#多个键
left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left,right,on = ["key1","key2"],how = "outer")


# In[13]:


print(pd.merge(left,right,on = "key1"))
print(pd.merge(left,right,on = "key1",suffixes=["_left","_right"])) # suffixes 为冲突字段加区分字段


# In[16]:


left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
print(left1)
right1


# In[15]:


pd.merge(left1,right1,left_on="key",right_index=True)


# In[17]:


lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
lefth


# In[18]:


righth


# In[19]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)


# In[20]:


left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                 columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])


# In[21]:


left2


# In[22]:


right2


# In[23]:


pd.merge(left2,right2,how="outer",left_index=True,right_index=True)


# In[24]:


left2.join(right2,how="outer")


# In[25]:


another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
left2.join([right2, another])


# In[26]:


# 轴向连接
arr = np.arange(12).reshape((3,4))
print(arr)
np.concatenate([arr,arr],axis = 1)


# In[27]:


s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])

pd.concat([s1, s2, s3])


# In[28]:


pd.concat([s1,s2,s3],axis = 1)


# In[29]:


df5 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df6 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])
print(df5)
print(df6)


# In[30]:


pd.concat([df5,df6],axis = 1,keys = ["level1","level2"])


# In[31]:


pd.concat([df5, df6], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])


# In[32]:


df7 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df8 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
print(df7,"\n",df8)


# In[33]:


pd.concat([df7,df8],ignore_index=True)


# In[34]:


pd.concat([df7,df8])


# In[35]:


a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
print(a)
print(b)


# In[36]:


np.where(pd.isnull(a),b,a)


# In[37]:


b[:-2].combine_first(a[2:])


# In[38]:


#– Stack：将数据的列“旋转”为行
# – Unstack：将数据的行“旋转”为列


data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
print(data,"\n",
data.duplicated(),
data.drop_duplicates())


# In[39]:


###利用函数或映射进行数据转换
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[40]:


meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
data["animals"] = data["food"].map(str.lower).map(meat_to_animal)
data


# In[41]:


###替换值
data = Series([1., -999., 2., -999., -1000., 3.])
data


# In[42]:


data.replace([-999,-1000],np.nan)


# In[43]:


data.replace([-999, -1000], [np.nan, 0])


# In[44]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

bins = [18, 25, 35, 60, 100]
cuts = pd.cut(ages,bins)
cuts


# In[45]:


pd.value_counts(cuts)


# In[46]:


group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)


# In[48]:


###检测和过滤异常值
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
print(data.describe())

col = data[3]
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)]

data[np.abs(data) > 3] = np.sign(data) * 3 # sign是－取-1和+取一的函数
data.describe()


# In[51]:


mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(u'F:/电影/练数成金/第6周/data/movies.dat', sep='::', header=None,
                        names=mnames)
movies[:10]

