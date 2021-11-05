import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px

# https://insights.stackoverflow.com/survey/2021

path = '/Users/alf/Downloads/developer_survey_2021/'
data_df = pd.read_csv(path + 'survey_results_public.csv')

# path = '/Users/alf/Downloads/developer_survey_2020/'
# data_df = pd.read_csv(path + 'survey_results_public.csv')

# Handling null value - for 2020 data
print(data_df['Hobbyist'].isnull().sum())
data_df['Hobbyist'].fillna('NA', inplace=True)

# Pyplot to analyse the data

hobbyist_data_group = data_df.groupby('Hobbyist').count().reset_index()
plt.bar(hobbyist_data_group['Hobbyist'], hobbyist_data_group['Respondent'], label='Plot by Hobbyist')
plt.tight_layout()
plt.legend()

# plot using seaborn
sns.catplot(data=data_df, x='Hobbyist', kind='count')

plt.show()

#  Calculating the percentages of
yes_count = 0
no_count = 0
na_count = 0

for i in data_df.index:
    if data_df['Hobbyist'][i] == 'Yes':
        yes_count += 1
    elif data_df['Hobbyist'][i] == 'No':
        no_count += 1
    elif data_df['Hobbyist'][i] == 'NA':
        na_count += 1

tot_count = yes_count + no_count + na_count
yes_pct = (yes_count/tot_count) * 100
yes_pct = round(yes_pct, 2)
no_pct = (no_count/tot_count) * 100
no_pct = round(no_pct, 2)
na_pct = (na_count/tot_count) * 100
na_pct = round(na_pct, 2)

# Output of the print
# Yes: 78.17% & count is 50388
# No: 21.76% & count is 14028
# NA: 0.07% & count is 45

plt.plot(yes_pct, no_pct, na_pct)
plt.tight_layout()
plt.show()

# Analyzing which age group has the higher counts for 2020 data.

def age_df(num):
    if num < 18:
        return 'Under 18 Years old'
    elif 18 <= num <= 24:
        return '18-24 years old'
    elif 25 <= num <= 34:
        return '25-34 years old'
    elif 35 <= num <= 44:
        return '35-44 years old'
    elif 45 <= num <= 54:
        return '45-54 years old'
    elif 55 <= num <= 64:
        return '55-64 years old'
    elif num >= 65:
        return '65 years or older'
    else:
        return 'Prefer not to say'


total_count_age = 0

for ind in data_df['Age']:
    total_count_age += 1

data_df['Age_group'] = data_df['Age'].apply(age_df)

data_df_pct = data_df.groupby('Age_group').count().reset_index()

data_df_pct['Age_group_pct'] = 0
for i, ind in enumerate(data_df_pct['Respondent']):
    data_df_pct['Age_group_pct'][i] = (ind/total_count_age)*100

sns.catplot(data=data_df_pct, x='Age_group', y='Age_group_pct', legend=True, kind='bar',
            order=['Under 18 Years old', '18-24 years old', '25-34 years old', '35-44 years old', '45-54 years old',
                   '55-64 years old', '65 years or older', 'Prefer not to say'], palette='pastel')


plt.show()

total_age_counter = 0

for ind in data_df['ResponseId']:
    total_age_counter += 1


data_df_2021_age = data_df.groupby('Age1stCode').count().reset_index()


def age_pct(value):
    return round((value / total_age_counter) * 100, 2)


data_df_2021_age['Age_Pct'] = data_df_2021_age['ResponseId'].apply(age_pct)

# Using plotly
fig = px.bar(data_frame=data_df_2021_age, x='Age_Pct', y='Age1stCode', color='Age1stCode',
             labels={'Age1stCode': 'Age Group', 'Age_Pct': 'Percentage of Response'},
             category_orders={'Age1stCode': ['Younger than 5 years', '5 - 10 years', '11 - 17 years', '18 - 24 years',
                                             '25 - 34 years', '35 - 44 years', '45 - 54 years', '55 - 64 years',
                                             'Older than 64 years']})

fig.show()

data_df_age_group = data_df.groupby('Age_group').count().reset_index
plt.bar(data_df_age_group['Age_group'], data_df_age_group['Respondent'])
plt.show()

data_df['LanguageWorkedWith'].fillna('NA', inplace=True)

# lang_pd = data_df['LanguageWorkedWith']
lang_counter = Counter()

total_lang_counter = 0

for ind in data_df['LanguageWorkedWith']:
    lang_counter.update(ind.split(';'))
    total_lang_counter += 1

#  empty list for languages and popularity
languages = []
popularity = []

for lang in lang_counter.most_common(10):
    languages_pct = round((lang[1]/total_lang_counter) * 100, 2)
    languages.append(lang[0])
    popularity.append(languages_pct)
    # print(f'{lang}: {languages_pct}%')

lang_df = DataFrame(languages, columns=['languages'])
lang_df['popularity'] = popularity
print(total_lang_counter)
sns.catplot(data=lang_df, x=languages, y=popularity, kind='bar')

plt.show()

# Checking the categories of Educational Level
data_df_edlevel = data_df.groupby('EdLevel').count().reset_index()

sns.barplot(data=data_df_edlevel, y='EdLevel', x='Respondent', palette='pastel', orient='h', y_testwrap=20)
plt.show()

edLevel_counter = 0
for ind in data_df['ResponseId']:
    edLevel_counter += 1


def cal_pct_edlevel(value):
    return round((value / edLevel_counter) * 100, 2)


data_df_edlevel['edLevel_pct'] = data_df_edlevel['ResponseId'].apply(cal_pct_edlevel)

# #  using plotly
fig = px.bar(data_frame=data_df_edlevel, x='edLevel_pct', y='EdLevel', color='EdLevel', orientation='h',
             labels={'edLevel_pct': 'Percentage # of Response', 'EdLevel': 'Educational Level'})
fig.show()
