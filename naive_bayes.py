import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
class naive_bayes:
    
    # Read our dataset using read_csv()
    bbc_text = pd.read_csv('bbc-text.txt')
    bbc_text=bbc_text.rename(columns = {'text': 'News_Headline'}, inplace = False)
    bbc_text.head()
    bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
    bbc_text.category.unique()

    X = bbc_text.News_Headline
    y = bbc_text.category
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
    vector = CountVectorizer(stop_words = 'english',lowercase=False)
    # fit the vectorizer on the training data
    vector.fit(X_train)
    vector.vocabulary_
    X_transformed = vector.transform(X_train)
    X_transformed.toarray()
    # for test data
    X_test_transformed = vector.transform(X_test)
    from sklearn.naive_bayes import MultinomialNB
    naivebayes = MultinomialNB()
    naivebayes.fit(X_transformed, y_train)
