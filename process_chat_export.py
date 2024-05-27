import json
import pandas as pd
import re
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Convert data to dict
with open(r'data/input/chat_export.json',
          'r',
          encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create DataFrame: df with only 'from' and 'text' columns, grouped by 'from'
df = pd.DataFrame(data['messages'])[
    ['from',
     'text']].dropna(axis=0,
                     how='any'
                     ).groupby('from')

# Create empty DataFrame: messages_df to iterate over when adding grouped data
messages_df = pd.DataFrame()

# Loop through text in df to merge all text from each user
for key, item in df:
    messages = []
    for n in range(len(item)):
        messages.append(str(item['text'].iloc[n]))
    messages_df[key] = pd.Series(str(messages))

# Create empty DataFrame: text to iterate over when adding grouped data
text = pd.DataFrame()

# Extract only cyrillic words with regex
for name, data in messages_df.items():
    sent = data.values
    words = re.findall(r'[а-яА-Яё]+',
                       str(sent))
    text[name] = [word_tokenize(*[' '.join(words).lower()],
                                language='russian')]

# Generate a set of russian stop words
stop_words = set(stopwords.words('russian'))

# Create empty DataFrame: filtered to filter out stop words from each user
filtered = dict()

# Filter out stop words from each user's texts with a loop
for name, data in text.items():
    filtered[name] = pd.DataFrame([word for word in data[0]
                                   if word not in stop_words])
    print(name, 'filtering... done')

frequency = dict()

# Plot word frequency distribution for each user's filtered text
for name, data in filtered.items():
    fdist = FreqDist(data[0])
    frequency[name] = pd.DataFrame(fdist.most_common(500),
                                   columns=['word',
                                            'count'])

# Create plotly figure
fig = go.Figure()
for k, v in frequency.items():
    fig.add_trace(
        go.Bar(y=v['word'],
               x=v['count'],
               name=k,
               orientation='h'))

fig['layout']['yaxis']['autorange'] = 'reversed'
fig.update_layout(barmode='stack')

# Write figure to an .html file
fig.write_html('data/output/chat_export_stats.html')
