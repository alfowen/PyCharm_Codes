import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import oil CSV into a dataframe

pd_oil = pd.read_csv('/Users/alf/Downloads/oil.csv')

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
avg_sales = pd_oil.groupby('date').mean('sales').reset_index()
avg_sales['sales_ewm'] = pd_oil['sales'].ewm(span=7, adjust=False).mean()

ax1 = avg_sales.plot(x='date', y=['sales', 'sales_ewm'], figsize=(18, 6))
plt.title('Sales over a period of time')
plt.show()
# print(pd_oil.head())
