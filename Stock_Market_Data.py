# Import Libraries
from datetime import datetime
import datetime
import requests
import pandas as pd
import csv
from io import StringIO
import psycopg2 as ps
from cryptography.fernet import Fernet
import time

# from datetime import datetime
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

# Encryption Key
file = open('/Users/alf/PycharmProjects/pythonProject/fernet_key', 'rb')
key = file.read()
file.close()
f = Fernet(key)

# Read Input Parameter and store them in a list
file_list = []
file_open = open('/Users/alf/PycharmProjects/pythonProject/stock_input_file_parameter', 'rb')
for file_line in file_open:
    file_list.append(file_line)
file_open.close()
stock_symbol = 'AAPL'

# Read Stock symbol from a file
stock_list = []
file_stock = open('/Users/alf/PycharmProjects/pythonProject/Stock_Symbol', 'rb')
for file_line in file_stock:
    stock_list.append(file_line.rstrip())
file_stock.close()

# Decrypt the data using the key
host_name = f.decrypt(file_list[1])
dbname = 'database-1'
port = '5432'
username = f.decrypt(file_list[2])
password = f.decrypt(file_list[0])
user_agent = f.decrypt(file_list[3])
user_agent = user_agent.decode()
conn = None

# Dynamic date and time  and convert to unix timestamp
dt = datetime.datetime.now()
# dt = datetime.datetime.now() - datetime.timedelta(days=1)
unix_time = time.mktime(dt.timetuple())
unix_time = int(unix_time)

# Hidden API URL for Yahoo Finance
url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_symbol

# Input Parameters For the API call
headers = {'user-agent': user_agent}

params = {'period1': unix_time,
          'period2': unix_time,
          # 'range': '1d',
          'interval': '1d',
          'events': 'history'}


# Create Table
def create_table(curr):
    create_table_command = ("""CREATE TABLE IF NOT EXISTS yahoo_stocks (
                            stock_symbol VARCHAR(10) NOT NULL,
                            stock_date DATE NOT NULL,
                            stock_open NUMERIC(10,2) NOT NULL,
                            stock_high NUMERIC(10,2) NOT NULL,
                            stock_low NUMERIC(10,2) NOT NULL,
                            stock_adj_close NUMERIC(10,2) NOT NULL,
                            stock_volume INTEGER NOT NULL,
                            insert_update_ts TIMESTAMP NOT NULL );
                           """)
    curr.execute(create_table_command)


# Function to connect to AWS RDS. Host name, username and password are encrypted
def connect_to_db_table_creation(host_name, username, password, port):
    try:
        conn = ps.connect(host=host_name.decode(), user=username.decode(), password=password.decode(), port=port)
        curr = conn.cursor()
        create_table(curr)
        curr.close()
        conn.commit()
    except (Exception, ps.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
        print('Connected & Table Created!')
    return curr


# Function to call the API using request module. There is also data formatting using pandas
def call_api(stock_list, headers, params):
    for i, j in enumerate(stock_list):
        url = 'https://query1.finance.yahoo.com/v7/finance/download/' + j.decode()
        response = requests.get(url, headers=headers, params=params)
        web_data = StringIO(response.text)
        # Format data in using pandas/dataframe
        pd_stock_interim = csv.reader(web_data)
        pd_stock_inter = pd.DataFrame(pd_stock_interim)
        new_header = pd_stock_inter.iloc[0]
        pd_stock_inter.columns = new_header
        if i == 0:
            # Creating an empty dataframe use the columns names from the API and also add the stock symbol as a new column.
            new_header = pd_stock_inter.iloc[0]
            pd_stock_inter.columns = new_header
            new_header = [*new_header, *['Stock_Symbol', 'Insert_Update_TS']]
            pd_stock = pd.DataFrame(columns=new_header)
        pd_stock = pd_stock.append(pd_stock_inter[1:], ignore_index=True)
        pd_stock['Stock_Symbol'].iloc[i] = j.decode()
    pd_stock.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    pd_stock[['Open', 'High', 'Low', 'Close', 'Adj_Close']] = pd_stock[
        ['Open', 'High', 'Low', 'Close', 'Adj_Close']].astype(float)
    pd_stock[['Open', 'High', 'Low', 'Close', 'Adj_Close']] = pd_stock[
        ['Open', 'High', 'Low', 'Close', 'Adj_Close']].round(decimals=2)
    pd_stock['Insert_Update_TS'] = datetime.datetime.now()
    new_header = [*new_header, *['Stock_Symbol', 'Insert_Update_TS']]
    return pd_stock, new_header

# Function to check for change data capture
def change_data_capture(curr, Stock_Symbol, Date):
    query = ("""SELECT stock_symbol, stock_date FROM yahoo_stocks where stock_symbol = %s and stock_date = %s;""")
    var_select = (Stock_Symbol, Date)
    curr.execute(query, var_select)
    return curr.fetchone() is not None

# Function to update row if the data exist
def update_row(curr, Stock_Symbol, Date, Open, High, Low, Adj_Close, Volume, Insert_Update_TS):
    query = ("""UPDATE yahoo_stocks
                SET stock_open = %s,
                    stock_high = %s,
                    stock_low = %s,
                    stock_adj_close = %s,
                    stock_volume = %s,
                    insert_update_ts = %s
                WHERE stock_symbol = %s
                    and stock_date = %s;""")
    vars_update = (Open, High, Low, Adj_Close, Volume, Insert_Update_TS, Stock_Symbol, Date)
    curr.execute(query, vars_update)

# Function to prep data to be updated row by row
def update_db(curr, pd_stock):
    temp_df = pd.DataFrame(columns=new_header)
    temp_df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    for i, row in pd_stock.iterrows():
        if change_data_capture(curr, row['Stock_Symbol'], row['Date']):
            update_row(curr, row['Stock_Symbol'], row['Date'], row['Open'], row['High'], row['Low'],
                       row['Adj_Close'], row['Volume'], row['Insert_Update_TS'])
        else:
            temp_df = temp_df.append(row)
    return temp_df

# Function to insert row if there is no record already in the db
def insert_row(curr, Stock_Symbol, Date, Open, High, Low, Adj_Close, Volume, Insert_Update_TS):
    query_insert = ("""INSERT INTO yahoo_stocks (stock_symbol, stock_date, stock_open, stock_high, stock_low, stock_adj_close, 
                stock_volume, insert_update_ts) VALUES(%s, %s, %s, %s, %s, %s, %s, %s);""")
    row_insert = (Stock_Symbol, Date, Open, High, Low, Adj_Close, Volume, Insert_Update_TS)
    curr.execute(query_insert, row_insert)

# Function to prep data to be inserted row by row
def insert_db(curr, pd_stock_insert):
    for i, row in pd_stock_insert.iterrows():
        insert_row(curr, row['Stock_Symbol'], row['Date'], row['Open'], row['High'],
                   row['Low'], row['Adj_Close'], row['Volume'], row['Insert_Update_TS'])

# Connection to run the update & insert
def connect_to_db_update_insert(host_name, username, password, port):
    try:
        conn = ps.connect(host=host_name.decode(), user=username.decode(), password=password.decode(), port=port)
        curr = conn.cursor()
        pd_stock_insert = update_db(curr, pd_stock)
        if pd_stock_insert.empty:
            print('Update Is Successfully Completed')
        else:
            insert_db(curr, pd_stock_insert)
            print('Insert Is Successfully Completed')
        curr.close()
        conn.commit()
    except (Exception, ps.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Overall Process Completed With No Failure')
    return None

# # Connect and create table.
# connect_to_db_table_creation(host_name, username, password, port)

# Extract data using API and push the data to DB.
pd_stock, new_header = call_api(stock_list, headers, params)
print(pd_stock)
connect_to_db_update_insert(host_name, username, password, port)
