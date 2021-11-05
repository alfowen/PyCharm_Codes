import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt

# Import oil CSV into a dataframe
path = '/Users/alf/Downloads/'
pd_oil = pd.read_csv(path + 'oil.csv')

# Import oil train data into a dataframe
pd_train = pd.read_csv(path + 'train.csv')

# Import transaction data into a dataframe
pd_trans = pd.read_csv(path + 'transactions.csv')

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

# avg_sales = pd_train.groupby('date').agg({'sales': 'mean'}).reset_index()
# avg_sales['weekly_avg_sales'] = avg_sales['sales'].ewm(span=7, adjust=False).mean()

# Plot graph
# ax1 = avg_sales.plot(x='date',y=['sales', 'weekly_avg_sales'], figsize=(18,6))

# avg_transaction = pd_trans.groupby('date').agg({'transactions': 'mean'}).reset_index()
# avg_transaction['weekly_avg_transaction'] = avg_transaction['transactions'].ewm(span=7, adjust=False).mean()

# ax2 = avg_transaction.plot(x='date', y=['transactions', 'weekly_avg_transaction'], figsize=(18,6))

# Find the correlation between sales and transaction

# pd_oil['sales'] = avg_sales['sales']
# pd_oil['transaction'] = avg_transaction['transactions']

# sns.heatmap(pd_oil.corr(), annot=True)
# sns.lmplot(data=pd_oil, x='sales', y='transaction')

# Plot Density plot
# pd_oil['daily_change_sales'] = pd_oil['sales'].pct_change()
# sns.histplot(data=pd_oil['daily_change_sales'], kde=True, bins=10, color='red', stat='density')

# pd_oil['daily_change_transaction'] = pd_oil['transaction'].pct_change()
# sns.histplot(data=pd_oil['daily_change_transaction'], kde=True, bins=10, color='red', stat='density')

# Calculate P-Value using scipy
# r, p = stats.pearsonr(pd_oil['sales'], pd_oil['transaction'])
# print(round(r, 4), round(p, 4))

# Print correlation
# print(pd_oil.corr())
# plt.show()

#  Check what is the percentage spent on promotion on each of the family category
# pd_train_sum = pd_train.groupby('family').sum().reset_index()
# sns.histplot(data=pd_train_sum, x='family', y='sales')
# plt.show()
# pd_train_sum['pct_sales_onpromotion'] = pd_train_sum['onpromotion']/pd_train_sum['sales']*100
# pd_sorted = pd_train_sum.sort_values('pct_sales_onpromotion', ascending=False).reset_index()


# sns.histplot(data=pd_sorted.head(5), x='family', y='pct_sales_onpromotion')
# print(pd_sorted[:10])
# family = pd_train_sum['family']
# pct_sales_onpromotion = pd_train_sum['pct_sales_onpromotion']
# plt.barh(pd_sorted.family.head(10), pd_sorted.pct_sales_onpromotion.head(10))
# plt.xlabel('Number of people Who use')
# plt.ylabel('Most Popular Languages')
# plt.tight_layout()
# plt.show()

# print(pd_train_sum.sort_values('pct_sales_onpromotion',ascending=False))

# Check data at different point in time.

pd_train['date'] = pd.to_datetime(pd_train['date'])
pd_train['day_of_week'] = pd_train['date'].dt.day_of_week
pd_train['month'] = pd_train['date'].dt.month
pd_train['year'] = pd_train['date'].dt.year

# print(pd_train.head())

data_grouped_day = pd_train.groupby(['day_of_week']).mean()['sales']
data_grouped_month = pd_train.groupby('month').mean()['sales']
data_grouped_year = pd_train.groupby('year').mean()['sales']

# plt.plot(data_grouped_day, label='Sales by Day')
# function to make 0-6 to mon - sun
# def day_month_year(val):
#     if val == 0:
#         return 'Mon'
#     if val == 1:
#         return 'Tue'
#     if val == 2:
#         return 'Wed'
#     if val == 3:
#         return 'Thur'
#     if val == 4:
#         return 'Fri'
#     if val == 5:
#         return 'Sat'
#     if val == 6:
#         return 'Sun'

# Create a dataframe of the grouped data and reindex
data_grouped_day_df = DataFrame(data_grouped_day).reset_index()
data_grouped_month_df = DataFrame(data_grouped_month).reset_index()
data_grouped_year_df = DataFrame(data_grouped_year).reset_index()
# used a function and apply to map the key value pair
# data_grouped_day_df['day_in_words'] = data_grouped_day_df['day_of_week'].apply(day_month_year)
# print(data_grouped_day_df.head())

# Used Dict Map to lookup on key value pairs
days_words = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thur', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
data_grouped_day_df['day_in_words'] = data_grouped_day_df['day_of_week'].map(days_words)
print(data_grouped_day_df)

months_words = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
data_grouped_month_df['month_in_words'] = data_grouped_month_df['month'].map(months_words)

data_grouped_month_df.drop(['month'], axis=1, inplace=True)
data_grouped_day_df.drop(['day_of_week'], axis=1, inplace=True)
print(data_grouped_day_df.head())
print(data_grouped_month_df.head())

# Plot By Day
plt.subplots(3, 1, figsize=(20, 5))
plt.style.use('ggplot')
plt.subplot(131)
plt.bar(data_grouped_day_df.day_in_words, data_grouped_day_df.sales, color='red', label='Sales By Day')
plt.title('Sales By Day')
plt.xlabel('Day of week')
plt.ylabel('Sales')
plt.legend()

# Plot By Month
plt.subplot(132)
plt.bar(data_grouped_month_df['month_in_words'], data_grouped_month_df['sales'], color='purple', label='Sales By Month')
plt.title('Sales By Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()

# Plot By Year
plt.subplot(133)
plt.bar(data_grouped_year_df['year'], data_grouped_year_df['sales'], color='black', label='Sales By Year')
plt.title('Sales By Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()

plt.tight_layout()
plt.show()

