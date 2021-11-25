import pandas as pd
from pandas import DataFrame, Series
from plotly.subplots import make_subplots
import numpy as np
# Data Visualization Libraries
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style
from mlxtend.plotting import plot_confusion_matrix

# Import Modeling Libraries
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# Import Pickle
import pickle

desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

path = '/Users/alf/Downloads/data_science/'

heart_raw_df = pd.read_csv(path + 'heart_failure_clinical_records_dataset.csv')

# --------------------- Code to categorize data by age and plot the death event ---------------------

def age_group(age):
    if 0 <= age <= 10:
        return '0_to_10'
    elif 10 < age <= 20:
        return '11_to_20'
    elif 20 < age <= 30:
        return '21_to_30'
    elif 30 < age <= 40:
        return '31_to_40'
    elif 40 < age <= 50:
        return '41_to_50'
    elif 50 < age <= 60:
        return '51_to_60'
    elif 60 < age <= 70:
        return '61_to_70'
    elif 70 < age <= 80:
        return '71_to_80'
    elif 80 < age <= 90:
        return '81_to_90'
    elif 91 < age <= 95:
        return '91_to_95'
    else:
        return 'Older than 95'


heart_raw_df['age_category'] = heart_raw_df['age'].apply(age_group)
death_rate_age_cat = heart_raw_df.groupby(['age_category', 'sex']).sum()['DEATH_EVENT'].reset_index()
diab_age_cat = heart_raw_df.groupby(['age_category', 'sex']).sum()['diabetes'].reset_index()
bp_age_cat = heart_raw_df.groupby(['age_category', 'sex']).sum()['high_blood_pressure'].reset_index()
smoke_age_cat = heart_raw_df.groupby(['age_category', 'sex']).sum()['smoking'].reset_index()
anaemic_age_cat = heart_raw_df.groupby(['age_category', 'sex']).sum()['anaemia'].reset_index()

all_data_age_cat = pd.concat([death_rate_age_cat, diab_age_cat.iloc[:, 2:], bp_age_cat.iloc[:, 2:],
                              smoke_age_cat.iloc[:, 2:], anaemic_age_cat.iloc[:, 2:]],
                             axis=1, join='inner')
all_data_age_cat_male = all_data_age_cat[all_data_age_cat['sex'] == 1]
all_data_age_cat_female = all_data_age_cat[all_data_age_cat['sex'] == 0]

print(all_data_age_cat)
print(all_data_age_cat_female)
print(all_data_age_cat_male)

# Bar graph by Gender - for Women/Men
trace1 = go.Bar(x=all_data_age_cat['age_category'], y=all_data_age_cat['DEATH_EVENT'],
                name='Number Of Death By Age Category')
trace2 = go.Bar(x=all_data_age_cat['age_category'], y=all_data_age_cat['diabetes'],
                name='Number Of Diabetic By Age Category')
trace3 = go.Bar(x=all_data_age_cat['age_category'], y=all_data_age_cat['high_blood_pressure'],
                name='Number Of Blood Pressure By Age Category')
trace4 = go.Bar(x=all_data_age_cat['age_category'], y=all_data_age_cat['smoking'],
                name='Number Of Smokers By Age Category')
trace5 = go.Bar(x=all_data_age_cat['age_category'], y=all_data_age_cat['anaemia'],
                name='Number Of Anaemic By Age Category')
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(trace1)
fig.add_trace(trace2, secondary_y=True)
fig.add_trace(trace3, secondary_y=True)
fig.add_trace(trace4, secondary_y=True)
fig.add_trace(trace5, secondary_y=True)
fig['layout'].update(height=1000, width=1800, title='Plot Number of death By Various Other Factors For Women/Men',
                     xaxis=dict(
                         tickangle=-25
                     ))
fig.show()
# --------------------- Code to categorize data by age and plot the death event ---------------------

# ----------------------Scatter Plot To Check for Correlation ----------------------
# style.use('gpplot')
p = 'serum_creatinine'
plt.scatter(heart_raw_df[p], heart_raw_df['time'])
plt.xlabel(p)
plt.ylabel('Death Event By Health Conditions')
plt.show()
print(style.available)
# ----------------------Plot Predictions----------------------

# More Plotting to analyze the data
X_data = heart_raw_df[['anaemia', 'diabetes', 'high_blood_pressure', 'smoking',
                       'serum_creatinine', 'ejection_fraction', 'DEATH_EVENT']]
sns.heatmap(data=X_data.corr(), annot=True)
# Serum_creatinine shows closer correlation but the others are not so much
sns.jointplot(data=heart_raw_df, x='age', y='DEATH_EVENT', kind='scatter', color='seagreen')

# Distplot on Age
sns.histplot(data=heart_raw_df, x='age', kde=True, bins=100, color='purple', stat='density')
# You see more spread on 40 to 80 or more from 45 to 70

# Boxplot on Gender & Age
sns.boxplot(data=heart_raw_df, x='sex', y='age')
# Men are more spread on between lower 50 to 70 where as Women are more between 50 to upper 60
plt.show()

# -----------------Prediction Model Data Prep-----------------
predict = 'DEATH_EVENT'

X = np.array(heart_raw_df[['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'serum_creatinine']])
y = np.array(heart_raw_df[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


# -----------------Prediction Model Data Prep-----------------
# -----------------Prediction Model - Linear Regression Model -----------------
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc_lig = linear.score(x_test, y_test)

# Accuracy Score
print(f'Accuracy Of Linear Regression Model For Selected Condition is:{acc_lig*100}%')
# Coefficient Value
print(f'Coefficient: {linear.coef_}')
# Intercept Value
print(f'Intercept: {linear.intercept_}')

# Prediction on test data set.
prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(round(prediction[x]), x_test[x], y_test[x])

# Linear Regression on Anaemia, Diabetes, High Blood pressure, Smoking, Serum Creatinine  is 15.11 %
# -----------------Prediction Model - Linear Regression Model -----------------

log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
acc_log = log_reg.score(x_test, y_test)

print(f'Accuracy Of Logistic Model For Selected Condition is: {acc_log*100}%')
# -----------------Prediction Model - Logistic Regression Model -----------------
# -----------------Prediction Model - SVC Regression Model -----------------
svm_svc = SVC()
svm_svc.fit(x_train, y_train)
svm_svc_pred = svm_svc.predict(x_test)
acc_svc = svm_svc.score(x_test, y_test)
print(f'Accuracy Of SVC Model For Selected Condition is: {acc_svc*100}%')
# -----------------Prediction Model - SVC Regression Model -----------------
# -- The Target Label Is A Classification problem Not A Regression, All The Regression Model Accuracy Are Very Low --

# -----------------Prediction Model - KNN Classification Model -----------------
best = 0
for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    acc_snn = knn.score(x_test, y_test)
    if acc_snn > best:
        best = acc_snn
        print(f'Accuracy Of KNN Model For Selected Condition is: {acc_snn*100}%')
cm = confusion_matrix(y_test, knn_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title('KNN Model - Confusion Matrix')
plt.xticks(range(2), ['Survived', 'Death'], fontsize=16)
plt.yticks(range(2), ['Survived', 'Death'], fontsize=16)
plt.show()
# KNN shows accuracy of model up to 86% when checked for 100 times.
# -----------------Prediction Model - KNN Classification Model -----------------


# ----------------Pickle The Model Output ----------------
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# ----------------------Pickle The Highest Accuracy rate----------------------
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    log_reg = LogisticRegression()

    log_reg.fit(x_train, y_train)
    log_reg_pred = log_reg.predict(x_test)
    acc_log = log_reg.score(x_test, y_test)

    print(f'Accuracy Of Logistic Model For Selected Condition is: {acc_log*100}%')

    if acc_log > best:
        best = acc_log
        with open('/Users/alf/Downloads/data_science/Heart_Failure.pickle', 'wb') as f:
            pickle.dump(log_reg, f)
pickle_in = open('/Users/alf/Downloads/data_science/Heart_Failure.pickle', 'rb')
log_reg = pickle.load(pickle_in)
# ----------------------Pickle The Highest Accuracy rate----------------------


