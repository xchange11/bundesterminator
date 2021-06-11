import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def get_missing_party(speeches_df, bio_data_df):
    '''If party name is missing, try grab it from bio data.

    This code is extremely slow!
    If you know a better way, please update it.'''
    for index, speech in speeches_df.iterrows():
        try:
            # Check if there's a NaN party
            if np.isnan(speech['party']):
                int_speaker_id = int(speech['speaker_id'])
                # Grab party from bio data and add the missing party to the speeches df
                party = bio_data_df[(bio_data_df['person_id'] == int_speaker_id
                                     )]['party'].ravel()[0]
                speeches_df.at[index, 'party'] = party

        except:
            pass

    return speeches_df


def party_simplifier(df):
    '''Replaces the long party names with nice and short ones'''
    party_repl_dict = {
        'Fraktion der Sozialdemokratischen Partei Deutschlands':
        'SPD',
        'Fraktion der Freien Demokratischen Partei':
        'FDP',
        'Fraktion der Christlich Demokratischen Union/Christlich - Sozialen Union':
        'CDU/CSU',
        'BÜNDNIS\xa090/DIE GRÜNEN':
        'BÜNDNIS 90/DIE GRÜNEN',
        'Bündnis 90/Die Grünen':
        'BÜNDNIS 90/DIE GRÜNEN',
        'fraktionslos':
        'Fraktionslos',
        'SPD':
        'SPD',
        'FDP':
        'FDP',
        'CDU/CSU':
        'CDU/CSU',
        'BÜNDNIS 90/DIE GRÜNEN':
        'BÜNDNIS 90/DIE GRÜNEN',
        'Fraktionslos':
        'Fraktionslos',
        'AfD':
        'AfD',
        'DIE LINKE':
        'DIE LINKE',
        'Bremen':
        'Bremen'
    }
    df.party = df.party.map(party_repl_dict)

    return df


def impute_party(speeches_df, bio_data_df):
    '''Imputes most of the missing/nan party affiliations'''
    df = get_missing_party(speeches_df, bio_data_df)
    df = party_simplifier(df)
    return df


# Text processing
def del_punct(text):
    '''Remove punctuation'''
    punct = string.punctuation
    additional_punct = \
        '❝❞❛❜‘’‛‚“”„‟«»‹›Ꞌ"<>@×‧¨․꞉:⁚⁝⁞‥…⁖⸪⸬⸫⸭⁛⁘⁙⁏;⦂⁃‐‑‒-–⎯—―_~⁓⸛⸞⸟ⸯ¬/\⁄\⁄'\
        '|⎜¦‖‗†‡·•⸰°‣⁒%‰‱&⅋§÷+±=꞊′″‴⁗‵‶‷‸*⁑⁎⁕※⁜⁂!‼¡?¿⸮⁇⁉⁈‽⸘¼½¾²³©®™℠℻'\
        '℅℁⅍℄¶⁋❡⁌⁍⸖⸗⸚⸓()[]\{\}⸨⸩❨❩❪❫⸦⸧❬❭❮❯❰❱❴❵❲❳⦗⦘⁅⁆〈〉⏜⏝⏞⏟⸡⸠⸢⸣⸤⸥⎡⎤⎣⎦⎨⎬⌠⌡'\
        '⎛⎠⎝⎞⁀⁔‿⁐‾⎟⎢⎥⎪ꞁ⎮⎧⎫⎩⎭⎰⎱'
    punctuation = ''.join(list(set(punct + additional_punct)))
    return ''.join([c for c in text if c not in punctuation])


def to_lowercase(text):
    '''Lowercase the text'''
    return text.lower()


def remove_numbers(text):
    '''Remove all numbers'''
    return ''.join([c for c in text if not c.isdigit()])


def remove_umlauts(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'Ae')
    text = text.replace('Ö', 'Oe')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')
    return text


def remove_stopwords(text):
    '''Stopwords, ciao'''
    wordlist = text.split()
    stop_words = stopwords.words('german')
    stop_words = [remove_umlauts(x) for x in stop_words]
    return ' '.join([w for w in wordlist if w not in stop_words])


def lemmatize(text):
    '''Get the root of the word'''
    wordlist = text.split()
    lemmy = WordNetLemmatizer()
    return ' '.join([lemmy.lemmatize(w) for w in wordlist])


def listify(text):
    '''Convert the input text to a list. Each word becomes a list element'''
    return text.split() if type(text) == str else []


def remove_non_party(df):
    df_removed = df[(df.party != "Fraktionslos") & (df.party != "Bremen")]
    return df_removed


def balance(data, treshold=500_000):
    min_speech_number = data.party.value_counts().min()

    if (min_speech_number <= treshold):
        treshold = min_speech_number

    cdu_speeches_truncted = data[data.party == "CDU/CSU"][:treshold]
    afd_speeches_truncted = data[data.party == "AfD"][:treshold]
    spd_speeches_truncted = data[data.party == "SPD"][:treshold]
    fdp_speeches_truncted = data[data.party == "FDP"][:treshold]
    linke_speeches_truncted = data[data.party == "DIE LINKE"][:treshold]
    green_speeches_truncted = data[data.party ==
                                   "BÜNDNIS 90/DIE GRÜNEN"][:treshold]
    frames = [
        cdu_speeches_truncted, afd_speeches_truncted, spd_speeches_truncted,
        fdp_speeches_truncted, linke_speeches_truncted, green_speeches_truncted
    ]
    df = pd.concat(frames)
    # they are now all ordered by party, so we shuffle before returning
    df = df.sample(frac=1)
    return df


def basic_preprocess(text):
    return listify(
        lemmatize(
            remove_stopwords(
                remove_numbers(to_lowercase(remove_umlauts(
                    del_punct(text)))))))
