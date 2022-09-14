import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = 'Post', 'Emotion', 'Sentiment'

# Prepare the binary contents of the file for the json loader
file = gzip.open('goemotions.json.gz', 'rb')
entries = json.load(file)

df = pd.DataFrame(entries, columns=labels)

# group all the emotion and sentiment columns to in 
# dictionaries with the category and its count
emotions = df.pivot_table(columns=labels[1], aggfunc='size')
sentiments = df.pivot_table(columns=labels[2], aggfunc='size')

fig, (emo, sent) = plt.subplots(1, 2)

# Emotions Pie Chart
emo.set_title("Emotions")
emo.pie(emotions.values, labels=emotions.keys(),
        shadow=False, startangle=90, rotatelabels=True)

# Sentiment Pie Chart
sent.set_title("Sentiment")
sent.pie(sentiments.values, labels=sentiments.keys(),autopct='%1.1f%%',
        shadow=False, startangle=90)
plt.show()
