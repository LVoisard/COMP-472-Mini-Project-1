from calendar import c
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection.tests import test_split
from sklearn.naive_bayes import MultinomialNB

labels = 'Post', 'Emotion', 'Sentiment'


def main():
    df = parse_dataset()

    # group all the emotion and sentiment columns to dictionaries with the category and its count
    sentences = df[labels[0]]

    # Print to analyse dataset
    # draw_dataset_analysis(emotions=emotions, sentiments=sentiments)

    process_dataset(df)


def process_dataset(dataframe: pd.DataFrame):

    post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split(
        dataframe[labels[0]].astype(str).tolist(),
        dataframe[labels[1]].astype(str).tolist(),
        test_size=0.20)

    le = LabelEncoder()
    Y = le.fit_transform(sentiment_train)

    post_vectorizer = CountVectorizer()
    X = post_vectorizer.fit_transform(post_train)
    post_vectorizer.get_feature_names_out()

    clf = MultinomialNB()
    clf.fit(X, Y)

    X2 = post_vectorizer.transform(post_test)
    predic = clf.predict(X2)

    df = pd.DataFrame(le.inverse_transform(predic))
    sent_predic = df.pivot_table(columns=0, aggfunc='size')

    fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%',
             shadow=False, startangle=90)
    plt.show()



def parse_dataset():
    """Prepare the binary contents of the file for the json loader.
    Returns the dataframe with of the file contents"""
    file = gzip.open('goemotions.json.gz', 'rb')
    entries = json.load(file)
    return pd.DataFrame(entries, columns=labels)


def draw_dataset_analysis(dataframe):
    """Organize the dataset so the user may analyse it.
    The function outputs a diagram with two pie charts"""
    emotions = dataframe.pivot_table(columns=labels[1], aggfunc='size')
    sentiments = dataframe.pivot_table(columns=labels[2], aggfunc='size')

    fig, (emo, sent) = plt.subplots(1, 2)

    # Emotions Pie Chart
    emo.set_title("Emotions")
    emo.pie(emotions.values, labels=emotions.keys(),
            shadow=False, startangle=90, rotatelabels=True)

    # Sentiment Pie Chart
    sent.set_title("Sentiment")
    sent.pie(sentiments.values, labels=sentiments.keys(), autopct='%1.1f%%',
             shadow=False, startangle=90)
    plt.show()


main()
