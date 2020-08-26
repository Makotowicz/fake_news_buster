#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import numpy as np

def import_merge_df(df_1_path,df_2_path):

    '''Import DataFrames and merge them, adding true/false encodings'''

    data_fake = pd.read_csv(df_1_path)
    data_true = pd.read_csv(df_2_path)

    # data_fake = pd.read_csv(df_1)
    # data_true = pd.read_csv(df_2)


    data_fake["true/false"] = 1
    data_fake["true/false_description"] = "false"

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


def Data_Cleaning(df_1_path, df_2_path):
    '''Delete useless rows (with https..in every column)
        and adjust datetime object'''

    #Call merge/import function
    data_concat_reset_index = import_merge_df(df_1_path,df_2_path)


    #Filter out wrong "https"-values
    list_indexes_to_drop = data_concat_reset_index.query('date.str.contains("https")').index
    data = data_concat_reset_index.drop(data_concat_reset_index.index[list_indexes_to_drop])

    #Convert date to datetimeobjects
    data["date"] = data["date"].map(try_parsing_date)

    return data


def Drop_words_helper_fct(x, word_list):
    for word in word_list:
        x = x.replace(word, "")
    return x


def Drop_words(df, column_name, word_list):
    new_column_name = f'cleaned_{column_name}'
    df[new_column_name] = df[column_name].apply(lambda x: Drop_words_helper_fct(x,word_list))
    return df



