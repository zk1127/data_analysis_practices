
# coding: utf-8

# In[1]:


from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import numpy.random as randn
import matplotlib.pyplot as plt


# In[2]:


from datetime import datetime
now = datetime.now()
now


# In[3]:


now.year,now.month,now.day


# In[4]:


from datetime import timedelta
start = datetime(2017,11,12)
des = start + timedelta(12);
des


# In[5]:


#字符串转日期
stamp = datetime(2011,8,9)
str_day1 = str(stamp)
day1 = stamp.strftime("%Y-%m-%d")
str_day1


# In[6]:


value = "2011-8-9"
datetime.strptime(value,"%Y-%m-%d")


# In[7]:


datestrs = ["7/6/2011","5/6/2012"]
[datetime.strptime(x,"%m/%d/%Y") for x in datestrs]


# In[8]:


from dateutil.parser import parse
parse('2011-1-1')


# In[9]:


parse('Jan 30,1997 10:34 PM')


# In[10]:


parse('1/2/2001',dayfirst = True)


# In[11]:


pd.to_datetime(datestrs)


# In[12]:


from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6),index = dates)
ts


# In[13]:


stamp = ts.index[2]
ts[stamp]


# In[14]:


ts["20110102"]


# In[15]:


ts["20110106":"2011-01-12"]


# In[16]:


ts.truncate(after="20110106")
#切片


# In[17]:


dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.ix['5-2001']


# In[18]:


dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                          '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
grouped = dup_ts.groupby(level=0)
print(grouped.count())
grouped.mean()


# In[19]:


print(ts)
ts.resample("D").mean()


# In[20]:


index = pd.date_range("4/1/2012","6/1/2012")
index


# In[21]:


pd.date_range(start = "4/1/2012",periods = 20)
# pd.date_range(end = "4/1/2012",periods = 20)


# In[22]:


pd.date_range("4/1/2012","12/1/2012",freq="BM")


# In[23]:


pd.date_range("4/1/2012","12/1/2012",freq="1h30min")


# In[24]:


ts.shift(1,"D")


# In[25]:


ts.shift(1,freq = "90T")


# In[26]:


from pandas.tseries.offsets import Day,MonthEnd
now = datetime(2011,2,1)
now + MonthEnd(2)


# In[27]:


offset = MonthEnd()
# offset.rollforward(now)
offset.rollback(now)


# In[28]:


ts = Series(np.random.randn(20),index = pd.date_range("1/5/2011",periods = 20,freq = "4d"))
ts


# In[29]:


ts.groupby(offset.rollforward).mean()


# In[30]:


ts.resample("M",how = "mean")


# In[31]:


inputfile =open(u'F:/电影/练数成金/第12周/stock_px.csv')
close_px_all = pd.read_csv(inputfile, parse_dates=True, index_col=0)
close_px_all[:10]


# In[32]:


close_px = close_px_all.resample('B',fill_method="ffill").ffill()
close_px.info()


# In[33]:


close_px['AAPL'].plot()


# In[34]:


close_px.ix['2009'].plot()


# In[35]:


close_px.ix['01/2009':'12/2009'].plot()


# In[36]:


# 转换为季度
app_q = close_px["AAPL"].resample("Q-DEC",fill_method="ffill").ffill()
app_q.plot()


# In[37]:


close_px = close_px_all.asfreq('B').fillna(method = 'ffill').ffill()
pd.rolling_mean(close_px.AAPL,20).plot()


# In[38]:


spx_px = close_px_all['SPX']
# 平滑百分比
spx_rets = spx_px / spx_px.shift(1) - 1  
returns = close_px.pct_change()
corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr.plot()


# In[47]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                         figsize=(12, 7))
aapl_px = close_px.AAPL['2005':'2009']
ma60 = Series.rolling(aapl_px, 60, min_periods=50).mean()
ewma60 = Series.ewm(aapl_px, span=60).mean()

aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')


# In[51]:


spx_px = close_px_all['SPX']
spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
corr = pd.rolling_corr(returns.AAPL,spx_rets,125,min_periods= 50)
corr.plot()


# In[52]:


corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr.plot()


# In[54]:


## 时序案例分析
inputfile = u'F:\\电影\\练数成金\\第12周\\arima_data.xls'
arima_data = pd.read_excel(inputfile,index_col=u'日期')
arima_data = DataFrame(arima_data,dtype = np.float64)
arima_data


# In[55]:


plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正确显示中文
plt.rcParams['axes.unicode_minus'] = False # 用来正确显示负号
arima_data.plot()
plt.show()


# In[57]:


#自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(arima_data)


# In[58]:


from statsmodels.tsa.stattools import adfuller as adf
print(adf(arima_data[u'销量']))


# In[59]:


D_arima_data = arima_data.diff().dropna()
D_arima_data.columns = [u'时间差分']
D_arima_data.plot()


# In[60]:


plot_acf(D_arima_data)


# In[64]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_arima_data) #偏自相关图
adf(D_arima_data[u'时间差分'])#平稳性检测


# In[65]:


from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(D_arima_data,lags = 1)


# In[66]:


from statsmodels.tsa.arima_model import ARIMA
#定阶
pmax = int(len(D_arima_data)/10) #一般阶数不超过length/10
qmax = int(len(D_arima_data)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(arima_data, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)


# In[73]:


bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：{0}、{1}' .format(p,q))


# In[68]:


model = ARIMA(arima_data, (0,1,1)).fit() #建立ARIMA(0, 1, 1)模型


# In[69]:


model.summary() #给出一份模型报告


# In[70]:


model.forecast(5) #作为期5天的预测，返回预测结果、标准误差、置信区间。

