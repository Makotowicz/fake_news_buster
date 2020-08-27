

'''characters in text'''
def no_chracters(text):
    for i in text:
        fake_charac = len(text)
    return fake_charac

def no_characters_df(df, column):
    new_column_name = f'{column}_no_characters'
    df[new_column_name] = df[column].apply(no_chracters)


'''add column with number of spelling mistakes per text'''
def spellchecker(text):
    for i in text:
        spell = SpellChecker()
        misspelled = spell.unknown(text)
        return len(misspelled)

