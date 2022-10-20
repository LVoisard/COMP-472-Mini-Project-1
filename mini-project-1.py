# from calendar import c
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm, datasets
# from sklearn import metrics

# 1.1. Load Dataset
import gzip
import json
import pandas as pd
# 1.3. Plotting the distribution
import matplotlib.pyplot as plt
# 2.1. Displaying dataset tokens
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
# 2.2. Splitting the dataset
from sklearn.model_selection.tests import test_split
# 2.3. Classifier Training / Testing
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
# 2.4. Performance Classification Report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# 3.1 Word2Vec import
from gensim.downloader import load
# 3.2. Tokenizer
from nltk.tokenize import word_tokenize
# 3.3. Post Embedding
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
# 3.7. Performance Classification Report
from sklearn.metrics import classification_report

labels = 'Post', 'Emotion', 'Sentiment'


def main():

    # 1.1. Load Dataset ----------------------------
    dataset = parse_dataset()

    # 1.3. Data Extraction -------------------------
    # Group all the emotion and sentiment columns to dictionaries with the category and its count
    posts = dataset[labels[0]]
    emotions = dataset[labels[1]]
    sentiments = dataset[labels[2]]
    emotions_distribution = dataset.pivot_table(columns=labels[1], aggfunc='size')
    sentiments_distribution = dataset.pivot_table(columns=labels[2], aggfunc='size')

    # Plotting the distributions
    fig, (emo, sent) = plt.subplots(1, 2)
    # Emotions Pie Chart
    emo.set_title("Emotions")
    emo.pie(emotions_distribution.values, labels=emotions_distribution.keys(),
                shadow=False, startangle=90, rotatelabels=True)

    # Sentiment Pie Chart
    sent.set_title("Sentiment")
    sent.pie(
        sentiments_distribution.values, labels=sentiments_distribution.keys(),
        autopct='%1.1f%%', shadow=False, startangle=90)
    plt.show()

    # 2.1. Displaying dataset tokens ---------------
    post_vectorizer = CountVectorizer()
    X = post_vectorizer.fit_transform(dataset[labels[0]])
    print('There are', X.shape[1], 'unique tokens')
    emotions_label_encoder = LabelEncoder()
    Y = emotions_label_encoder.fit_transform(emotions)
    sentiments_label_encoder = LabelEncoder()
    Z = sentiments_label_encoder.fit_transform(sentiments)

    # 2.2. Splitting the dataset -------------------
    post_train, post_test, emotion_train, emotion_test, sentiment_train, sentiment_test = test_split.train_test_split(
        X, Y,  Z, test_size=0.20)

    # Print to analyse dataset
    # draw_dataset_analysis(dataset)

    base_mnb_emotion, base_mnb_sentiment = base_mnb(dataset, post_train, emotion_train, sentiment_train)    # 2.3.1.
    base_dt_emotion, base_dt_sentiment = base_dt(dataset, post_train, emotion_train, sentiment_train)       # 2.3.2.
    base_mlp_emotion, base_mlp_sentiment = base_mlp(dataset, post_train, emotion_train, sentiment_train)    # 2.3.3.
    top_mnb_emotion, top_mnb_sentiment = top_mnb(dataset, post_train, emotion_train, sentiment_train)       # 2.3.4.
    top_dt_emotion, top_dt_sentiment = top_dt(dataset, post_train, emotion_train, sentiment_train)          # 2.3.5.
    top_mlp_emotion, top_mlp_sentiment = top_mlp(dataset, post_train, emotion_train, sentiment_train)       # 2.3.6.

    # 2.5. Classification Report -------------------
    performance_report_1(emotion_test, sentiment_test, post_test, base_mnb_emotion, base_mnb_sentiment, base_dt_emotion, base_dt_sentiment,
                         base_mlp_emotion, base_mlp_sentiment, top_mnb_emotion, top_mnb_sentiment, top_dt_emotion, top_dt_sentiment,
                         top_mlp_emotion, top_mlp_sentiment, emotions_label_encoder, sentiments_label_encoder)

    # 3.1. Word2Vec import -------------------------
    model = load('word2vec-google-news-300')

    # 3.2. Tokenizer -------------------------------
    posts_vec_train, posts_vec_test = tokenizer(dataset)

    # 3.3. Post Embedding --------------------------
    posts_vec_train, posts_vec_test, post_embedding_train, post_embedding_test= post_embedding(posts_vec_train, posts_vec_test, model)

    # 3.4. Hit Rate --------------------------------
    hit_rate(posts_vec_train, posts_vec_test, model)

    # 3.5. Base MLP --------------------------------
    w2v_base_mlp_emotion = MLPClassifier()
    w2v_base_mlp_emotion.fit(post_embedding_train, emotion_train)

    w2v_base_mlp_sentiment = MLPClassifier()
    w2v_base_mlp_sentiment.fit(post_embedding_train, sentiment_train)

    # 3.6. Top MLP ---------------------------------
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'solver': ['adam', 'sgd']
    }
    w2v_top_mlp_emotion = GridSearchCV(MLPClassifier(max_iter=1), param_grid)
    w2v_top_mlp_emotion.fit(post_embedding_train, emotion_train)

    w2v_top_mlp_sentiment = GridSearchCV(MLPClassifier(max_iter=1), param_grid)
    w2v_top_mlp_sentiment.fit(post_embedding_train, sentiment_train)

    # 3.7. Performance Classification Report -------
    performance_report_2(emotion_test, sentiment_test, w2v_base_mlp_emotion, w2v_base_mlp_sentiment,
                         w2v_top_mlp_emotion, w2v_top_mlp_sentiment, post_embedding_test,
                         emotions_label_encoder, sentiments_label_encoder)


def base_mnb(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.1.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split( X, Y, test_size=0.2)
    # clf = MultinomialNB()
    # clf.fit(post_train, sentiment_train)
    # predic = clf.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%', shadow=False, startangle=90)
    # plt.show()
    print("Start BASE_MNB")
    base_mnb_emotion = MultinomialNB()
    base_mnb_emotion.fit(post_train, emotion_train)

    base_mnb_sentiment = MultinomialNB()
    base_mnb_sentiment.fit(post_train, sentiment_train)
    print("End BASE_MNB")
    return base_mnb_emotion, base_mnb_sentiment


def base_dt(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.2.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split( X, Y, test_size=0.2)
    # clf = DecisionTreeClassifier()
    # clf.fit(post_train, sentiment_train)
    # predic = clf.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%',  shadow=False, startangle=90)
    # plt.show()
    print("Start BASE_DT")
    base_dt_emotion = DecisionTreeClassifier()
    base_dt_emotion.fit(post_train, emotion_train)

    base_dt_sentiment = DecisionTreeClassifier()
    base_dt_sentiment.fit(post_train, sentiment_train)
    print("End BASE_DT")
    return base_dt_emotion, base_dt_sentiment


def base_mlp(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.3.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split( X, Y, test_size=0.2)
    # mlpc = MLPClassifier(max_iter=5)
    # mlpc.fit(post_train, sentiment_train)
    # predic = mlpc.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%', shadow=False, startangle=90)
    # plt.show()
    print("Start BASE_MLP")
    base_mlp_emotion = MLPClassifier(max_iter=3)
    base_mlp_emotion.fit(post_train, emotion_train)

    base_mlp_sentiment = MLPClassifier(max_iter=3)
    base_mlp_sentiment.fit(post_train, sentiment_train)
    print("End BASE_MLP")
    return base_mlp_emotion, base_mlp_sentiment


def top_mnb(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.4.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    #
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    #
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split(
    #     X,
    #     Y,
    #     test_size=0.2)
    #
    # param_grid = {'alpha': [0, 0.25, 0.5, 0.75]}
    # clfCV = GridSearchCV(MultinomialNB(), param_grid)
    # print(clfCV.fit(post_test, sentiment_test))
    # predic = clfCV.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%', shadow=False, startangle=90)
    # plt.show()
    print("Start TOP_MNB")
    param_grid = {'alpha': [0, 0.25, 0.5, 0.75]}
    top_mnb_emotion = GridSearchCV(MultinomialNB(), param_grid)
    top_mnb_emotion.fit(post_train, emotion_train)

    top_mnb_sentiment = GridSearchCV(MultinomialNB(), param_grid)
    top_mnb_sentiment.fit(post_train, sentiment_train)
    print("End TOP_MNB")
    return top_mnb_emotion, top_mnb_sentiment


def top_dt(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.5.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split( X, Y, test_size=0.2)
    # param_grid = {
    #     'criterion': ['entropy'],
    #     'max_depth': [2, 8],
    #     'min_samples_split': [2, 4, 6]
    # }
    # clfCV = GridSearchCV(DecisionTreeClassifier(), param_grid)
    # print(clfCV.fit(post_test, sentiment_test))
    # predic = clfCV.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%', shadow=False, startangle=90)
    # plt.show()
    print("Start TOP_DT")
    param_grid = {
        'criterion': ['entropy'],
        'max_depth': [2, 8],
        'min_samples_split': [2, 4, 6]
    }
    top_dt_emotion = GridSearchCV(DecisionTreeClassifier(), param_grid)
    top_dt_emotion.fit(post_train, emotion_train)

    top_dt_sentiment = GridSearchCV(DecisionTreeClassifier(), param_grid)
    top_dt_sentiment.fit(post_train, sentiment_train)
    print("End TOP_DT")
    return top_dt_emotion, top_dt_sentiment


def top_mlp(dataframe: pd.DataFrame, post_train, emotion_train, sentiment_train):  # 2.3.6.
    # le = LabelEncoder()
    # Y = le.fit_transform(dataframe[labels[1]].astype(str).tolist())
    # post_vectorizer = CountVectorizer()
    # X = post_vectorizer.fit_transform(dataframe[labels[0]].astype(str).tolist())
    # post_vectorizer.get_feature_names_out()
    # post_train, post_test, sentiment_train, sentiment_test = test_split.train_test_split(X,Y,test_size=0.2)
    # param_grid = {
    #     'activation': ['logistic', 'tanh', 'relu', 'identity'],
    #     'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    #     'solver': ['adam', 'sgd']
    # }
    # clfCV = GridSearchCV(MLPClassifier(max_iter=3), param_grid)
    # print(clfCV.fit(post_test, sentiment_test))
    # predic = clfCV.predict(post_test)
    # df = pd.DataFrame(le.inverse_transform(predic))
    # sent_predic = df.pivot_table(columns=0, aggfunc='size')
    # fig = plt.pie(sent_predic.values, labels=sent_predic.keys(), autopct='%1.1f%%', shadow=False, startangle=90)
    # plt.show()
    print("Start TOP_MLP")
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30,50),(10,10,10)],
        'solver': ['adam', 'sgd']
    }
    top_mlp_emotion = GridSearchCV(MLPClassifier(max_iter=3), param_grid)
    top_mlp_emotion.fit(post_train, emotion_train)

    top_mlp_sentiment = GridSearchCV(MLPClassifier(max_iter=3), param_grid)
    top_mlp_sentiment.fit(post_train, sentiment_train)
    print("End TOP_MLP")
    return top_mlp_emotion, top_mlp_sentiment


def performance_report_1(emotion_test, sentiment_test, post_test, base_mnb_emotion, base_mnb_sentiment, base_dt_emotion,
                         base_dt_sentiment, base_mlp_emotion, base_mlp_sentiment, top_mnb_emotion, top_mnb_sentiment, top_dt_emotion,
                         top_dt_sentiment, top_mlp_emotion, top_mlp_sentiment, emotions_label_encoder, sentiments_label_encoder):  # 2.4.
    # 2.5.
    print("Start PERFORMANCE_REPORT_1")
    print('Base MNB\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, base_mnb_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, base_mnb_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, base_mnb_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, base_mnb_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))

    print('Base DT\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, base_dt_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, base_dt_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, base_dt_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, base_dt_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))

    print('Base MLP\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, base_mlp_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, base_mlp_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, base_mlp_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, base_mlp_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))

    print('Top MNB\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, top_mnb_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, top_mnb_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, top_mnb_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, top_mnb_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))

    print('Top DT\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, top_dt_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, top_dt_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, top_dt_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, top_dt_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))

    print('Top MLP\nEmotion\nConfusion Matrix \n')
    print(confusion_matrix(emotion_test, top_mlp_emotion.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(emotion_test, top_mlp_emotion.predict(post_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\nConfusion Matrix \n')
    print(confusion_matrix(sentiment_test, top_mlp_sentiment.predict(post_test)))
    print('\n Classification Report \n')
    print(classification_report(sentiment_test, top_mlp_sentiment.predict(post_test), target_names=sentiments_label_encoder.classes_))
    print("End PERFORMANCE_REPORT_1")


def tokenizer(dataset):  # 3.2
    print("Start TOKENIZER")
    postsVec = dataset[labels[0]].apply(word_tokenize)
    posts_vec_train, posts_vec_test = test_split.train_test_split(postsVec, test_size=0.20)
    words_train = 0
    for i, post in enumerate(posts_vec_train):
        for word in post:
            words_train = words_train + 1

    print('There are', len(posts_vec_train), 'sentences')
    print('There are', words_train, 'tokens')
    print('Training only')
    print("End TOKENIZER")
    return posts_vec_train, posts_vec_test


def post_embedding(posts_vec_train, posts_vec_test, model):  # 3.3.
    print("Start POST_EMBEDDING")
    post_embedding_train = np.zeros((len(posts_vec_train), model.vector_size))
    for i, post in enumerate(posts_vec_train):
        post_vec = np.zeros((model.vector_size,))
        words = 0
        for word in post:
            if word in model:
                words = words + 1
                post_vec = np.add(post_vec, model[word])
        if words == 0:
            words = 1
        post_embedding_train[i] = np.divide(post_vec, words)

    post_embedding_test = np.zeros((len(posts_vec_test), model.vector_size))
    for i, post in enumerate(posts_vec_test):
        post_vec = np.zeros((model.vector_size,))
        words = 0
        for word in post:
            if word in model:
                words = words + 1
                post_vec = np.add(post_vec, model[word])
        if words == 0:
            words = 1
        post_embedding_test[i] = np.divide(post_vec, words)

    print(post_embedding_train)
    print("End POST_EMBEDDING")
    return posts_vec_train, posts_vec_test, post_embedding_train, post_embedding_test


def hit_rate(posts_vec_train, posts_vec_test, model):
    hits = 0
    total_words = 0
    for i, post in enumerate(posts_vec_train):
        for word in post:
            total_words = total_words + 1
            if word in model:
                hits = hits + 1
    print('Training hit rate', hits / total_words)

    hits = 0
    total_words = 0
    for i, post in enumerate(posts_vec_test):
        for word in post:
            total_words = total_words + 1
            if word in model:
                hits = hits + 1
    print('Testing hit rate', hits / total_words)


def performance_report_2(emotion_test, sentiment_test,
                         w2v_base_mlp_emotion, w2v_base_mlp_sentiment,
                         w2v_top_mlp_emotion, w2v_top_mlp_sentiment,
                         post_embedding_test, emotions_label_encoder,
                         sentiments_label_encoder):
    print('Base MLP\nEmotion\n Classification Report \n')
    print(classification_report(emotion_test, w2v_base_mlp_emotion.predict(post_embedding_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\n Classification Report \n')
    print(classification_report(sentiment_test, w2v_base_mlp_sentiment.predict(post_embedding_test), target_names=sentiments_label_encoder.classes_))

    print('Top MLP\nEmotion\n Classification Report \n')
    print(classification_report(emotion_test, w2v_top_mlp_emotion.predict(post_embedding_test), target_names=emotions_label_encoder.classes_))

    print('Sentiment\n Classification Report \n')
    print(classification_report(sentiment_test, w2v_top_mlp_sentiment.predict(post_embedding_test), target_names=sentiments_label_encoder.classes_))


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
