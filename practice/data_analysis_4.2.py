
# coding: utf-8

# In[1]:


###正则表达式
import re
text = "foo    bar\t baz  \tqux"
re.split('\s+', text)


# In[2]:


text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r"[A-Z0-9._%+-]+@[A-Z0-9._]+\.[A-Z]{2,4}"
# pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+|.[A-Z]{2,4}'
regex = re.compile(pattern,flags=re.IGNORECASE)
regex.findall(text)


# In[3]:


pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
m.groups()


# In[4]:


regex.findall(text)


# In[5]:


print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))


# In[6]:


import json
db = json.load(open(u"F:/电影/练数成金/第6周/data/foods.json"))
len(db)


# In[7]:


db[0].keys()


# In[8]:


db


# In[10]:


import pandas as pd
from pandas import Series,DataFrame
nutrients = DataFrame(db[0]['nutrients'])
nutrients[:7]


# In[11]:


info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)
info


# In[13]:


pd.value_counts(info.group)


# In[14]:


nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

nutrients


# In[15]:


nutrients.duplicated().sum()


# In[16]:


nutrients.drop_duplicates


# In[17]:


col_mapping = {'description' : 'food',
               'group'       : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info


# In[18]:


col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients


# In[20]:


ndata = pd.merge(info,nutrients,on = "id",how = "outer")
ndata


# In[21]:


ndata.ix[1000]


# In[24]:


import matplotlib.pyplot as plt
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind='barh')


# In[35]:


# 1. 读入  肝气郁结证型系数.xls  数据集，将数据集按照等距、小组等量 两种方式 分别分为5组数据，分别计算5组数据的中位数与标准差
path = u"F:/电影/练数成金/第6周/第6周作业数据/"
# inputfile = open(path+"BHP1.csv")
df1 = pd.read_excel(path +"肝气郁结证型系数.xls ",header = None,skiprows=1)
df1.columns =["values"]
df1


# In[38]:


df1['Group_XZDJ']=pd.cut(df1['values'],5,precision=2) #将值列按等距方式分为5组并赋值新列  
df1['Group_XZDL']=pd.qcut(df1['values'], 5, precision=2) #将值列按等距等量方式分为5组并赋值新列  
group_xzdl=df1['values'].groupby(df1['Group_XZDJ'])   #将值列按等距分组列准备数据  
group_xzdj=df1['values'].groupby(df1['Group_XZDL'])   #将值列按等距等量分组列准备数据  
print(group_xzdj.median(),"\n",group_xzdj.std(),"\n",
     group_xzdl.median(),"\n",group_xzdl.std())


# In[42]:


# 2. 读入BHP1.csv，使用适当的方法填补缺失值
from scipy.interpolate import lagrange
inputfile = open(path + "BHP1.csv")
df2 = pd.read_csv(inputfile)
# 计算拉格朗日值
def ployinterp_column(s, n, k=3):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) # 最后的括号就是我们要插值的n
#逐个元素判断是否需要插值
for i in df2.columns:
  for j in range(len(df2)):
    if (df2[i].isnull())[j]: #如果为空即插值。
      df2[i][j] = ployinterp_column(df2[i], j)
df2
# 3. 读入BHP2.xlsx，与BHP1数据集合并为BHP数据集
# 4. 将BHP数据集中的成交量（volume）替换为 high、median、low 三种水平（区间自行定义）


# In[44]:


# 3. 读入BHP2.xlsx，与BHP1数据集合并为BHP数据集
df3 = pd.read_excel(path+"BHP2.xlsx")
df3


# In[48]:


df4 = pd.merge(df2,df3,how="outer")
df4


# In[55]:


# 4. 将BHP数据集中的成交量（volume）替换为 high、median、low 三种水平（区间自行定义）
volume = df4['volume']  
  
idx = 0  
for i in volume:  
    if i >= 3000000 and i < 4000000:  
        df4['volume'][idx] = "median"  
    elif i >= 4000000 :  
        df4['volume'][idx] = "high"  
    else:   
        df4['volume'][idx] = "low "  
    idx+=1     
df4  

