
def import_merge_df(df_1,df_2):
    '''Import DataFrames and merge them, adding true/false encodings'''

    data_fake = pd.read_csv(f'fake_news_buster/data/{df_1}')
    data_true = pd.read_csv(f'fake_news_buster/data/{df_2}')

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


def Data_Cleaning():
    '''Delete useless rows (with https..in every column)
        and adjust datetime object'''

    #Call merge/import function
    data_concat_reset_index = import_merge_df("Fake.csv","True.csv")


    #Filter out wrong "https"-values
    list_indexes_to_drop = data_2.query('date.str.contains("https")').index
    data = data_concat_reset_index.drop(data_2.index[list_indexes_to_drop])

    #Convert date to datetimeobjects
    data["date"] = data["date"].map(try_parsing_date)

    return data





