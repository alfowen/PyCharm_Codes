import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import datetime as dt
from os import listdir
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

path = '/Users/alf/Downloads/data_science/Sales_data/'

file_list = [file for file in listdir(path)]

# Read all files into one dataframe

sales_all_data = DataFrame()
for ind in file_list:
    df = pd.read_csv(path + ind)
    sales_all_data = pd.concat([sales_all_data, df])

sales_all_data.dropna(how='all', inplace=True)

#  Clean-Up data on NaN & Junk data.
sales_all_data.dropna(how='any', inplace=True)
sales_all_data = sales_all_data[sales_all_data['Order Date'].str[0:2] != 'Or']

# ---------------------------- Block of code for Sales by Month Analysis.----------------------------
sales_all_data['Order Date'] = pd.to_datetime(sales_all_data['Order Date'])

sales_all_data.reset_index()

sales_all_data['order_date_month'] = sales_all_data['Order Date'].dt.month
sales_all_data['Quantity Ordered'] = pd.to_numeric(sales_all_data['Quantity Ordered'])
sales_all_data['Price Each'] = pd.to_numeric(sales_all_data['Price Each'])
sales_all_data['Sales'] = sales_all_data['Quantity Ordered'] * sales_all_data['Price Each']
sales_all_data_by_month = sales_all_data.groupby('order_date_month').sum().reset_index()

# Using plotly to plot a bar graph
fig = px.bar(data_frame=sales_all_data_by_month, x='order_date_month', y='Sales', color='Sales',
             labels={'order_date_month': 'Month Number', 'Sales': 'Sales in USD ($)'})
fig.show()
# ---------------------------- Block of code for Sales by Month Analysis.----------------------------

# ---------------------------- Block of code for Sales by city Analysis.----------------------------
def city_name(value):
    return value.split(',')[1]


def state_name(value):
    return value.split(',')[2].split(' ')[1]


# sales_all_data['City_Name'] = sales_all_data['Purchase Address'].apply(lambda x: city_name(x) + '-' + state_name(x))

# Using f string
sales_all_data['City_Name'] = sales_all_data['Purchase Address'].apply(lambda x: f'{city_name(x)}-{state_name(x)}')
sales_all_data['Quantity Ordered'] = pd.to_numeric(sales_all_data['Quantity Ordered'])
sales_all_data['Price Each'] = pd.to_numeric(sales_all_data['Price Each'])
sales_all_data['Sales'] = sales_all_data['Quantity Ordered'] * sales_all_data['Price Each']
sales_by_city = sales_all_data.groupby('City_Name').sum().reset_index()
fig = px.bar(data_frame=sales_by_city, x='City_Name', y='Sales', color='Sales',
             labels={'City_Name': 'City with State Name', 'Sales': 'Sales in USD($)'})
fig.show()

# San Fran tops the list in sales followed by LA and New York
# ---------------------------- Block of code for Sales by city Analysis.----------------------------

# ---------- What time should the display advertisement to maximize customer buying product ----------

sales_all_data['Order Date'] = pd.to_datetime(sales_all_data['Order Date'])
sales_all_data['Order_Date_Hour'] = sales_all_data['Order Date'].dt.hour
sales_all_data['Order_Date_Min'] = sales_all_data['Order Date'].dt.minute
sales_all_data['Quantity Ordered'] = pd.to_numeric(sales_all_data['Quantity Ordered'])
sales_all_data['Price Each'] = pd.to_numeric(sales_all_data['Price Each'])
sales_all_data['Sales'] = sales_all_data['Price Each'] * sales_all_data['Quantity Ordered']
sales_by_hour = sales_all_data.groupby('Order_Date_Hour').sum().reset_index()

# fig = px.line(data_frame=sales_by_hour, x='Order_Date_Hour', y='Sales', title='Sales By Hour',
#               labels={'Order_Date_Hour': 'Sales By Hours', 'Sales': 'Sales in USD($)'})
# fig.show()

# Using tick mode in plotly

fig = go.Figure(go.Scatter(
    x=sales_by_hour['Order_Date_Hour'],
    y=sales_by_hour['Sales'],
))

fig.update_layout(
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1
    )
)
fig.show()

# ---------- What time should the display advertisement to maximize customer buying product ----------

# ------------------------ Block of code to check what product sold in combination with others------------------------
sales_all_dup = sales_all_data[sales_all_data['Order ID'].duplicated(keep=False)]
sales_all_dup['Group_Data'] = sales_all_dup.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
sales_all_dup = sales_all_dup[['Order ID', 'Group_Data']].drop_duplicates()

order_count = Counter()

for ind in sales_all_dup['Group_Data']:
    order_count.update(combinations(ind.split(','), 2))

for key, value in order_count.most_common(10):
    print(key, value)
# ------------------------ Block of code to check what product sold in combination with others------------------------

# ------------------------ Block of code to check what product that sold the most------------------------
sales_all_data['Quantity Ordered'] = pd.to_numeric(sales_all_data['Quantity Ordered'])
sales_product_sold = sales_all_data.groupby('Product').sum()['Quantity Ordered'].reset_index()
sales_product_sold.rename(columns={'Quantity Ordered': 'Order_Count'}, inplace=True)

sales_all_data['Price Each'] = pd.to_numeric(sales_all_data['Price Each'])
sales_price_mean = sales_all_data.groupby('Product').mean()['Price Each'].reset_index()

trace1 = go.Bar(x=sales_product_sold['Product'], y=sales_product_sold['Order_Count'],
                name='Product by Number of Order Sold')
trace2 = go.Scatter(x=sales_price_mean['Product'], y=sales_price_mean['Price Each'],
                    name='Mean Price of each Product', yaxis='y2')

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(trace1)
fig.add_trace(trace2, secondary_y=True)
fig['layout'].update(height=1000, width=1800, title='Plot Between Mean Price vs Number of Product Sold', xaxis=dict(
    tickangle=-90
))

fig.show()
# ------------------------ Block of code to check what product that sold the most------------------------
