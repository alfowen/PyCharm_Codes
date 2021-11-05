import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

lang_df = pd.read_csv('/Users/alf/Downloads/data_lang.csv')
lang_response = lang_df['LanguagesWorkedWith']

lang_counter = Counter()

for i in lang_response:
    lang_counter.update(i.split(';'))
languages = []
popularity = []
for item in lang_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

languages.reverse()
popularity.reverse()

plt.barh(languages,popularity)
plt.xlabel('Number of people Who use')
plt.ylabel('Most Popular Languages')
plt.tight_layout()
plt.show()
