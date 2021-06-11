import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from bundestag.bundestrainer import Bundestrainer

# The system variable "Dyno" is only present in Heroku prod env
ON_PRODUCTION = 'DYNO' in os.environ

model_dir = ""
if ON_PRODUCTION == False:
    model_dir = "api/"

# Prepare DL model for prediction
bundestrainer = Bundestrainer()
bundestrainer.load_model(f'{model_dir}model2.tf')
bundestrainer.load_w2c(f'{model_dir}model2.w2v')
bundestrainer.load_party_mapping(f'{model_dir}party_mapping.txt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(speech_fragment, model=1):  # 1
    X = pd.Series([speech_fragment])

    pred = ""

    # Baseline Model
    if model == "1":
        # pipeline = get_model_from_gcp()
        pipeline = joblib.load(f'{model_dir}model.joblib')

        # make prediction
        results = pipeline.predict(X)

        # convert response from numpy to python type
        pred = results[0]

    # Random Generator
    elif model == "2":
        parties = [
                'CDU',
                'CSU',
                'SPD',
                'FDP',
                'LINKE',
                'GRÜNE',
                'PARTEI',
                'ÖDP',
                'PIRATEN',
            ]
        pred = random.choice(parties)

    elif model =="3":
        print(type(speech_fragment))
        pred = bundestrainer.predict_party_by_string(speech_fragment)


    return(dict(prediction=pred))
