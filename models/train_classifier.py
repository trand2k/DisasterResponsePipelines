import sys
import sqlalchemy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle
import nltk

def load_data(database_filepath):
    '''
    Load data
    Input:
        database_filepath: File path  sql 
    Output:
        X: Message data 
        Y: Categories 
        category_names: name categories
    '''
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df_mess = pd.read_sql_table('DisasterMessages', engine)
    X_train = df_mess['message']
    Y_train = df_mess.iloc[:, 4:]
    category_names = list(df_mess.columns[4:])

    return X_train, Y_train, category_names


def tokenize(text):
    '''
    Tokenize text
    Input:
        text: message text
    Output:
        lemmed: Tokenized text
    '''
    text_clean = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    tokenize = word_tokenize(text_clean)
    lemma = WordNetLemmatizer()

    tokens = []
    for tok in tokenize:
        clean_tok = lemma.lemmatize(tok).lower().strip()
        tokens.append(clean_tok)

    return tokens


def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        model 
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    params = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    model = GridSearchCV(pipeline, param_grid=params)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance 
    Input: 
        model: Model 
        X_test: Test data 
        Y_test: True lables
        category_names: name labels
    Output:
        None
    '''
    y_pred = model.predict(X_test)

    for idex,column in enumerate(category_names):
        print(column, classification_report(Y_test.values[:,idex],y_pred[:,idex]))


def save_model(model, model_filepath):
    '''
    Save model 
    Input: 
        model: 
        model_filepath:
    Output:
        None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()