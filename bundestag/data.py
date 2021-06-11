import os
import pandas as pd
import numpy as np


from bundestag.utils import del_punct, to_lowercase, remove_numbers, \
    remove_stopwords, lemmatize


# Temporary path until the data is hosted somewhere.
# Create a raw_data folder and dump the CSV in there.
DATA_SOURCE = os.path.abspath('raw_data')
GCP_BUCKET_NAME = "" # NEEDS TO BE PROVIDED HERE IN CODE
GCP_BUCKET_DATA_FOLDER = 'trained'


def get_data(nrows=1_000_000):
    '''Grab the data from somewhere'''
    csv_files = [
        'bio_data.csv',
        'speech_segments_a.csv',
        'speech_segments_b.csv',
    ]

    all_data = {}
    for csv in csv_files:
        key = os.path.splitext(csv)[0]
        csv_path = os.path.abspath(os.path.join(DATA_SOURCE, csv))
        all_data[key] = pd.read_csv(
            f"gs://{GCP_BUCKET_NAME}/{GCP_BUCKET_DATA_FOLDER}/{csv}",
            nrows=nrows,
            sep=';')

    all_data['speech_segments'] = all_data['speech_segments_a'].\
        append(all_data['speech_segments_b'])
    del all_data['speech_segments_a']
    del all_data['speech_segments_b']

    # format {'speech_segments': ..., 'bio_data': ...}
    return all_data


def clean_data(df):
    '''Data Cleanup'''
    clean_df = df.copy()
    clean_df = clean_df[['text', 'party']]
    # General cleanup. Consider this a baseline placeholder for now
    clean_df = clean_df.dropna()

    # Do text proprocessing. Replace as needed!
    clean_df.text = clean_df.text.map(del_punct)
    clean_df.text = clean_df.text.map(to_lowercase)
    clean_df.text = clean_df.text.map(remove_numbers)
    clean_df.text = clean_df.text.map(remove_stopwords)
    clean_df.text = clean_df.text.map(lemmatize)
    return clean_df


if __name__ == '__main__':
    df = get_data()
