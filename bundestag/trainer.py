from bundestag.data import get_data, clean_data
from bundestag.utils import impute_party
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import joblib
import os
from tempfile import mkdtemp
from google.cloud import storage



# Google Cloud Platform Data
GCP_BUCKET_NAME = "" # NEEDS TO BE PROVIDED HERE IN CODE
GCP_BUCKET_DATA_FOLDER = 'trained'


class Trainer():
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        preprocess = Pipeline([('vectorize', CountVectorizer())])

        pipe = Pipeline([('prep', preprocess),
                         ('multinb_model', MultinomialNB())])

        self.pipeline = pipe

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        '''evaluates the pipeline'''
        prediction = self.pipeline.predict(X_test)
        out = []
        for i, pred in enumerate(prediction):
            out.append(
                f'pred: {pred.ljust(25)} truth: {y_test.ravel()[i].ljust(25)}')
        # truth = y_test
        return '\n'.join(out[:30])
        # return y_test.ravel()

    def save_model(self, name):
        """Save the model into a .joblib format and upload to gcloud BUCKET!"""
        filename = os.path.join(name)
        joblib.dump(self.pipeline, filename)
        self.upload_model_to_gcp(filename)

    def upload_model_to_gcp(self, location):
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(location)
        blob.upload_from_filename(location)


if __name__ == '__main__':
    # Grab the data
    data = get_data()

    df = data['speech_segments']
    bio = data['bio_data']

    # Impute missing party values
    df = impute_party(df, bio)

    # Clean the data
    df = clean_data(df)

    # X and y
    X = df.drop('party', axis=1)
    y = df.party

    # Train Test Split
    X_train, X_test, y_train, y_test = \
        train_test_split(X.text, y, test_size=0.2)

    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()

    # Save model
    trainer.save_model('model.joblib')

    # evaluate
    # print(trainer.evaluate(X_test, y_test))
