import pandas as pd
from datetime import datetime
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from spellchecker import SpellChecker
import string
from sklearn.preprocessing import MinMaxScaler
import joblib
'''data cleaning'''
def try_parsing_date(text):
    if text != text:   ## pandas nan type not equal to itself
        return np.nan
    if "$date" in text:
        diff_time = text.replace("'$date", "")
        diff_time = diff_time.replace("': ", "")
        diff_time = diff_time.replace("{", "")
        diff_time = diff_time.replace("}", "")
        diff_time = int(diff_time)/1000
        date = datetime.utcfromtimestamp(diff_time)
        return date
    else:
        for fmt in ('%d-%b-%y', '%B %d, %Y', '%b %d, %Y','%b %d, %Y ','%B %d, %Y '):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    return np.nan
'''weekday one hot encoded'''
def weekday(day):
    weekday = day.dt.day_name()
    return weekday
#def ohe_weekday(df):
    #ohe = OneHotEncoder(sparse = False)
    #ohe.fit(df[['weekday']])
    #hair_length_oh = ohe.transform(df[['weekday']])
    #df["day_friday"],df["day_monday"],df['day_saturday'],df['day_sunday'],df['day_thursday'],df['day_tuesday'],df['day_wednesday'] = hair_length_oh.T
    #return df
def weekday_encode(row):
    row['day_sunday'] = 1 if row['weekday'] == 'Sunday' else 0
    row['day_friday'] = 1 if row['weekday'] == 'Friday' else 0
    row['day_monday'] = 1 if row['weekday'] == 'Monday' else 0
    row['day_tuesday'] = 1 if row['weekday'] == 'Tuesday' else 0
    row['day_thursday'] = 1 if row['weekday'] == 'Thursday' else 0
    row['day_wednesday'] = 1 if row['weekday'] == 'Wednesday' else 0
    row['day_saturday'] = 1 if row['weekday'] == 'Saturday' else 0
    return row
'''sentiment analysis _ 1'''
def get_polarity(x):
    x = TextBlob(x)
    return x.sentiment[0]
def get_subjectivity(x):
    x = TextBlob(x)
    return x.sentiment[1]
def feature_polarity_subjectivity(df, column):
    new_column_name_polarity = f'{column}_TextBlob_polarity_score'
    df[new_column_name_polarity] = df[column].apply(get_polarity)
    new_column_name_subjectivity = f'{column}_TextBlob_subjectivity_score'
    df[new_column_name_subjectivity] = df[column].apply(get_subjectivity)
    return df
'''sentiment analysis _ 2'''
def feature_vader_polarity_scores(df, column):
    new_column_name = f'{column}_Vader_negative_score'
    df[new_column_name] = df[column].apply(lambda x: vader.polarity_scores(x)["neg"])
    new_column_name = f'{column}_Vader_neutral_score'
    df[new_column_name] = df[column].apply(lambda x: vader.polarity_scores(x)["neu"])
    new_column_name = f'{column}_Vader_positive_score'
    df[new_column_name] = df[column].apply(lambda x: vader.polarity_scores(x)["pos"])
    new_column_name = f'{column}_Vader_compound_score'
    df[new_column_name] = df[column].apply(lambda x: vader.polarity_scores(x)["compound"])
    return df
'''length of articles'''
def no_chracters(text):
    for i in text:
          fake_charac = len(text)
    return fake_charac
def no_characters_df(df, column):
    new_column_name = f'{column}_no_characters'
    df[new_column_name] = df[column].apply(no_chracters)
    return df
'''punctuation ratio, Upper case letters ratio, numbers ratio'''
def character_ratiorizer(text):
    quotes = ['\"', '\"']
    quote_no = 0
    for symbol in text:
        if symbol in quotes:
            quote_no += 1
    return quote_no/len(text)
def is_upperizer(text):
    upper_no = 0
    for word in text:
        if word.isupper():
            upper_no += 1
    return upper_no/len(text)
def is_digiter(text):
    digit_no = 0
    for word in text:
        if word.isdigit():
            digit_no += 1
    return digit_no/len(text)
'''richness'''
def vocab_richnesser(text):
    tokens = word_tokenize(text)
    total_length = len(tokens)
    unique_words = set(tokens)
    unique_word_length = len(unique_words)
    try:
        return unique_word_length/total_length
    except ZeroDivisionError:
        return 0
''' typos count ratio'''
def preprocess_typos(text):
    text = text.replace(" t ", "'t ")
    text = text.replace(" t.", "'t.")
    text = text.replace(" t,", "'t,")
    text = text.replace(" t!", "'t!")
    text = text.replace(" t?", "'t?")
    text = text.replace(" s ", "'s ")
    text = text.replace(" s.", "'s.")
    text = text.replace(" s,", "'s,")
    text = text.replace(" s!", "'s!")
    text = text.replace(" s?", "'s?")
    text.split()
    for x in string.punctuation.replace("'", ""):
        text = text.replace(x, '')
    text = ''.join(word for word in text if not word.isdigit())
    return text
def typo_ratiorizer(text):
    spell = SpellChecker()
    misspells = spell.unknown(text)
    return len(misspells)/len(text)
'''scaler'''
def scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df.drop(['title', 'text', 'date'], axis=1))
    scaled_vals = scaler.transform(df.drop(['title', 'text','date'], axis=1))
    df[['day_friday',
        'day_monday',
        'day_saturday',
        'day_sunday',
        'day_thursday',
        'day_tuesday',
        'day_wednesday',
        'text_TextBlob_polarity_score',
        'text_TextBlob_subjectivity_score',
        'text_Vader_negative_score',
        'text_Vader_neutral_score',
        'text_Vader_positive_score',
        'text_Vader_compound_score',
        'title_TextBlob_polarity_score',
        'title_TextBlob_subjectivity_score',
        'title_Vader_negative_score',
        'title_Vader_neutral_score',
        'title_Vader_positive_score',
        'title_Vader_compound_score',
        'text_no_characters',
        'title_no_characters',
        'character_ratio',
        'upper_case_ratio',
        'numbers_ratio',
        'vocab_richness_text',
        'vocab_richness_title',
        'typo_ratio_text',
        'typo_ratio_title']]
    return scaled_vals
'''final call'''
def final_call(df):
  '''data cleaning'''
  #df["date"] = df["date"].apply(try_parsing_date)
  '''weekday'''
  df['weekday'] = df[['date']].apply(weekday)
  #df = ohe_weekday(df)
  df = df.apply(weekday_encode, axis = 1)
  '''sentiment analysis'''
  df = feature_polarity_subjectivity(df, 'text')
  df = feature_vader_polarity_scores(df, 'text')
  df = feature_polarity_subjectivity(df, 'title')
  df = feature_vader_polarity_scores(df, 'title')
  '''lenght of article'''
  df = no_characters_df(df, 'text')
  df = no_characters_df(df, 'title')
  '''punctuation ratio, Upper case letter ratio, numbers ratio'''
  df['character_ratio'] = df['title'].apply(character_ratiorizer)
  df['upper_case_ratio'] = df['title'].apply(is_upperizer)
  df['numbers_ratio'] = df['title'].apply(is_digiter)
  '''richness of vocab'''
  df['vocab_richness_text'] = df['text'].apply(vocab_richnesser)
  df['vocab_richness_title'] = df['title'].apply(vocab_richnesser)
  '''typos count'''
  df['preprocess_typo_text'] = df['text'].apply(preprocess_typos)
  df['preprocess_typo_title'] = df['title'].apply(preprocess_typos)
  df['typo_ratio_text'] = df['text'].apply(typo_ratiorizer)
  df['typo_ratio_title'] = df['title'].apply(typo_ratiorizer)
  '''drop colums'''
  df = df.drop(columns=['text', 'title', 'date', 'weekday', 'preprocess_typo_text', 'preprocess_typo_title'])
  X = df[['day_friday',
          'day_monday',
          'day_saturday',
          'day_sunday',
          'day_thursday',
          'day_tuesday',
          'day_wednesday',
          'text_TextBlob_polarity_score',
          'text_TextBlob_subjectivity_score',
          'text_Vader_negative_score',
          'text_Vader_neutral_score',
          'text_Vader_positive_score',
          'text_Vader_compound_score',
          'title_TextBlob_polarity_score',
          'title_TextBlob_subjectivity_score',
          'title_Vader_negative_score',
          'title_Vader_neutral_score',
          'title_Vader_positive_score',
          'title_Vader_compound_score',
          'text_no_characters',
          'title_no_characters',
          'character_ratio',
          'upper_case_ratio',
          'numbers_ratio',
          'vocab_richness_text',
          'vocab_richness_title',
          'typo_ratio_text',
          'typo_ratio_title']]
  '''scaler'''
  #df = scaler(df)
  scaling = joblib.load('final_scaler.gz')
  X = scaling.transform(X)
  '''import joblib of model and make prediction and return prediction'''
  loaded_model = joblib.load("Makotowicz_buzzfeed.sav")
  #result = loaded_model.score(X, y_test)
  result = loaded_model.predict_proba(X)
    #.iloc[0]])
  return result
