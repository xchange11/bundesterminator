import os
import pickle

from google.cloud import storage
from bundestag import data, utils
from bundestag.bundes_w2v import BundesW2V

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property



# Google Cloud Platform Data
GCP_BUCKET_NAME = "" # NEEDS TO BE PROVIDED HERE IN CODE
GCP_BUCKET_DATA_FOLDER = 'trained'

# MLFLOW server address
MLFLOW_URL = "" # NEEDS TO BE PROVIDED HERE IN CODE


class Bundestrainer():
    model = None
    loss = None
    optimizer = None
    metrics = None
    lstm_nodes = None
    keras_dense_layers = None
    last_layer_nodes = None
    batch_size = None
    patience = None
    epochs = None
    validation_split = None
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    speech_data = None
    bio_data = None
    balance_treshold = None
    w2v_model = None
    pad_len = None
    party_mapping = None

    def __init__(self,
                 loss="categorical_crossentropy",
                 optimizer="adam",
                 metrics=['accuracy'],
                 lstm_nodes=20,
                 keras_dense_layers={15: 'relu'},
                 last_layer_nodes=5,
                 batch_size=32,
                 patience=3,
                 epochs=10,
                 validation_split=0.3,
                 balance_treshold=500_000,
                 pad_len=300,
                 experiment_name=""): # NEEDS TO BE PROVIDED HERE IN CODE
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.lstm_nodes = lstm_nodes
        self.keras_dense_layers = keras_dense_layers
        self.last_layer_nodes = last_layer_nodes
        self.batch_size = batch_size
        self.patience = patience
        self.epochs = epochs
        self.validation_split = validation_split
        self.balance_treshold = balance_treshold
        self.pad_len = pad_len
        self.experiment_name = experiment_name

    def get_data(self):
        all_data = data.get_data()
        self.speech_data = all_data['speech_segments'][["text", "party",
                                                       "speech_id",
                                                       "speaker_id"]]
        self.bio_data = all_data['bio_data']

    def preprocess_dataframe(self):
        self.speech_data = utils.impute_party(self.speech_data, self.bio_data)
        self.speech_data = utils.remove_non_party(self.speech_data)
        self.speech_data = self.speech_data.dropna()
        self.speech_data["text"] = self.speech_data["text"].map(utils.basic_preprocess)
        self.speech_data = self.speech_data.dropna()
        self.speech_data = utils.balance(self.speech_data,
                                         self.balance_treshold)


    def prepare_data_for_training(self):
        self.X = self.speech_data["text"]
        self.y = self.speech_data["party"]

        self.encode_target() # labeling and cat generation
        self.split() # train-test-split and assign X_train etc. to instance prop
        self.init_w2v() # create w2v dict with X_train

        #Prepare X
        self.X_train = self.preprocess_text(self.X_train)
        self.X_test = self.preprocess_text(self.X_test)

    def preprocess_text(self, document_series):
        documents = document_series.to_list()
        documents = self.w2v_model.embedding(documents)
        documents = pad_sequences(documents,
                                  dtype='float32',
                                  padding='post',
                                  maxlen=self.pad_len)
        return documents

    def encode_target(self):
        party_df = pd.DataFrame()
        party_df["party"] = self.y
        party_df["party_encoded"] = LabelEncoder().fit_transform(
            party_df["party"])
        party_mapping = party_df.groupby("party_encoded").first()
        party_mapping = list(party_mapping["party"])
        self.party_mapping = party_mapping
        self.y = to_categorical(party_df["party_encoded"],
                                num_classes=len(party_mapping),
                                dtype="int32")

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def init_w2v(self):
        self.w2v_model = BundesW2V()
        self.w2v_model.init_model(self.X_train)

    def init_model(self):
        self.model = Sequential()
        self.model.add(layers.Masking())
        self.model.add(layers.LSTM(self.lstm_nodes, activation='tanh'))

        # Create dense layers based on user input.
        # Custom amount of neurons + different activation functions possible.
        for nodes, act in self.keras_dense_layers.items():
            self.model.add(layers.Dense(nodes, activation=act))

        # Try grabbing the correct number of last layer nodes
        # from the amount of present parties
        try:
            self.model.add(
                layers.Dense(len(self.party_mapping), activation='softmax'))
        except:
            self.model.add(
            layers.Dense(self.last_layer_nodes, activation='softmax'))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def fit_model(self):
        es = EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.model.fit(self.X_train,
                       self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split,
                       callbacks=[es])

        # MLFLOW Logging
        # Log the parameters
        self.mlflow_log_param('loss', self.loss)
        self.mlflow_log_param('optimizer', self.optimizer)
        self.mlflow_log_param('lstm_nodes', self.lstm_nodes)

        # Log the dense layers of the model
        for i, (nodes, act) in enumerate(self.keras_dense_layers.items(), 1):
            self.mlflow_log_param(f'dense_{i}_nodes', nodes)
            self.mlflow_log_param(f'dense_{i}_activation', act)

        self.mlflow_log_param('last_layer_nodes', self.last_layer_nodes)
        self.mlflow_log_param('batch_size', self.batch_size)
        self.mlflow_log_param('patience', self.patience)
        self.mlflow_log_param('epochs', self.epochs)
        self.mlflow_log_param('validation_split', self.validation_split)
        self.mlflow_log_param('balance_treshold', self.balance_treshold)
        self.mlflow_log_param('pad_len', self.pad_len)

        # Evaluate and log the metrics
        evaluation = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        for metric, value in zip(self.model.metrics_names, evaluation):
            try:
                self.mlflow_log_metric(metric, value)
            except:
                print(f"Metric :{metric} can't be logged. Does it even exist?")

    def get_init_fit(self):
        self.get_data()
        self.preprocess_dataframe()
        self.prepare_data_for_training()
        self.init_model()
        self.fit_model()

    def predict_party_by_string(self, text_string):
        processed_string = utils.basic_preprocess(text_string)
        processed_string_as_list = [processed_string]
        processed_string_as_series  = pd.Series(processed_string_as_list)
        vectorized_list = self.preprocess_text(processed_string_as_series)
        predicted_party_as_classes = self.model.predict_classes(vectorized_list)
        predicted_party_as_class = predicted_party_as_classes[0]
        predicted_party_as_string = self.party_mapping[predicted_party_as_class]
        return predicted_party_as_string

    def save_model(self, name):
        '''Save the trained model and upload to Google Cloud Platform'''
        filename = os.path.join(name)
        self.model.save(filename)
        self.upload_file_to_gcp(filename)

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def upload_file_to_gcp(self, location):
        '''Upload a file to the Google Cloud Platform'''
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(location)
        blob.upload_from_filename(location)

    def save_w2v(self, name):
        '''Save Word2vec model and also uplaod it to the Google Cloud'''
        filename = os.path.join(name)
        self.w2v_model.save(filename)
        self.upload_file_to_gcp(filename)

    def load_w2c(self, path):
        self.w2v_model = BundesW2V()
        self.w2v_model.load(path)

    def save_party_mapping(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.party_mapping, f)
        self.upload_file_to_gcp(path)

    def load_party_mapping(self, path):
        with open(path, "rb") as f:
            self.party_mapping = pickle.load(f)

    def save_speech_data(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.speech_data, f)

    def load_speech_data(self, path):
        with open(path, 'rb') as f:
            self.speech_data = pickle.load(f)

    # MLFLOW
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URL)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == '__main__':
    # Hire the trainer
    trainer = Bundestrainer()

    # Train those bastards
    trainer.get_init_fit()

    # Save the result
    trainer.save_w2v('model2.w2v')
    trainer.save_model('model2.tf')
    trainer.save_party_mapping('model2.pm')
