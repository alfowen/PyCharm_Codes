import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import oil CSV into a dataframe
path = '/Users/alf/Downloads/'
pd_oil = pd.read_csv(path+'oil.csv')

# Import oil train data into a dataframe
pd_train = pd.read_csv(path+'train.csv')

# Import transaction data into a dataframe
pd_trans = pd.read_csv(path+'transactions.csv')

# Rename column name
pd_oil.rename(columns={'dcoilwtico': 'sales'}, inplace=True)

# ax = pd_oil.set_index('date').plot(figsize=(16, 8))
# ax.set_xlabel('Date', fontsize='large')
# ax.set_ylabel('Crude Oil', fontsize='large')

# use seaborn to lineplot
# sns.lineplot(data=pd_oil,x='date',y='sales')
# plt.plot(figsize=(16, 8))
# plt.show()

# handle NaN values by Exponential Weighted
# avg_sales = pd_oil.groupby('date').mean('sales').reset_index()
# avg_sales['sales_ewm'] = pd_oil['sales'].ewm(span=7, adjust=False).mean()
#
# ax1 = avg_sales.plot(x='date', y=['sales', 'sales_ewm'], figsize=(18, 6))
# plt.title('Sales over a period of time')
# plt.show()

avg_sales = pd_train.groupby('date').agg({'sales': 'mean'}).reset_index()
avg_sales['weekly_avg_sales'] = avg_sales['sales'].ewm(span=7, adjust=False).mean()

# Plot graph
# ax1 = avg_sales.plot(x='date',y=['sales', 'weekly_avg_sales'], figsize=(18,6))

avg_transaction = pd_trans.groupby('date').agg({'transactions': 'mean'}).reset_index()
avg_transaction['weekly_avg_transaction'] = avg_transaction['transactions'].ewm(span=7, adjust=False).mean()

# ax2 = avg_transaction.plot(x='date', y=['transactions', 'weekly_avg_transaction'], figsize=(18,6))

# Find the correlation between sales and transaction

pd_oil['sales'] = avg_sales['sales']
pd_oil['transaction'] = avg_transaction['transactions']

# sns.heatmap(pd_oil.corr(), annot=True)
# sns.lmplot(data=pd_oil, x='sales', y='transaction')

# Plot Density plot
# pd_oil['daily_change_sales'] = pd_oil['sales'].pct_change()
# sns.histplot(data=pd_oil['daily_change_sales'], kde=True, bins=10, color='red', stat='density')

pd_oil['daily_change_transaction'] = pd_oil['transaction'].pct_change()
sns.histplot(data=pd_oil['daily_change_transaction'], kde=True, bins=10, color='red', stat='density')

# Calculate P-Value using scipy
r, p = stats.pearsonr(pd_oil['sales'], pd_oil['transaction'])
print(round(r, 4), round(p, 4))

# Print correlation
print(pd_oil.corr())
plt.show()


# print(pd_oil.head())



