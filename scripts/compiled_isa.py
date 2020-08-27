import pandas as pd
from datetime import datetime
import numpy as np

'''data cleaning'''

def import_merge_df(df_1_path_fake,df_2_path_true):
    '''Import DataFrames and merge them, adding true/false encodings'''
    data_fake = pd.read_csv(df_1_path_fake)
    data_true = pd.read_csv(df_2_path_true)
    # data_fake = pd.read_csv(df_1)
    # data_true = pd.read_csv(df_2)
    data_fake["true/false"] = 1
    data_fake["true/false_description"] = "fake"
    data_true["true/false"] = 0
    data_true["true/false_description"] = "true"
    data_concat = pd.concat([data_fake, data_true])
    data_concat_reset_index = data_concat.reset_index(drop=True)
    return data_concat_reset_index

def try_parsing_date(text):
    for fmt in ('%d-%b-%y', '%B %d, %Y', '%b %d, %Y','%b %d, %Y ','%B %d, %Y '):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return np.nan

def Data_Cleaning(df_1_path_fake, df_2_path_true):
    '''Delete useless rows (with https..in every column)
        and adjust datetime object'''
    #Call merge/import function
    data_concat_reset_index = import_merge_df(df_1_path_fake,df_2_path_true)
    #Filter out wrong "https"-values
    list_indexes_to_drop = data_concat_reset_index.query('date.str.contains("https")').index
    data = data_concat_reset_index.drop(data_concat_reset_index.index[list_indexes_to_drop])
    #Convert date to datetimeobjects
    data["date"] = data["date"].map(try_parsing_date)
    return data


'''sentiment analysis'''

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


'''length of articles'''

def no_chracters(text):
    for i in text:
          fake_charac = len(text)
    return fake_charac

def no_characters_df(df, column):
    new_column_name = f'{column}_no_characters'
    df[new_column_name] = df[column].apply(no_chracters)

''' punctuation ratio, Upper case letters ratio '''

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

''' richness '''

''' typos '''

''' final call of functions '''

df = Data_Cleaning(df_1_path_fake, df_2_path_true)
df = feature_polarity_subjectivity(df, 'text')
df = no_characters_df(df, 'text')

df('character_ratio') = character_ratiorizer(df['text'])
df('upper_case_ratio') = is_upperizer(df['text'])
df('numbers_ratio') = is_digiter(df['text'])



