# Bundesterminator
## What it is about?
The project explores ways to predict party affiliation by text segments.
Machine Learning and Deep Learning approaches are tested. This is the
result of a two-week project from the 
[Le Wagon](https://www.lewagon.com/de/berlin/data-science-course/full-time) 
Data Science Bootcamp, Batch 606 Berlin.

A demo is available at the following URL http://bundesterminator.herokuapp.com/.

## Data
For training the models the meeting minutes of the German
Parliament was used. They are available as XML files from the
[open data website](https://www.bundestag.de/services/opendata)
of the German Parliament. The XML files were pre-processed
and translated into CSV files (currently the python framework pandas
has no XML import).

## Folder Strucure
### api
The trained model can be exposed by a web API. It uses a lean setting based 
on [FastAPI](https://fastapi.tiangolo.com/) and 
[Uvicorn](https://www.uvicorn.org/). The deployment settings assume
a deployment on Heroku.

### bundestag
The `bundestag` folder represents the `bundestag` python package.
It contains the main files for training the models. 

#### trainer.py
Pipeline for a machine learning approach.

#### bundestrainer.py
Class to wrap functionalities to train a Deep Learning model with
[Tensorflow Keras](https://www.tensorflow.org/guide/keras/sequential_model)
 and a trained
[Gensim word2vev model](https://radimrehurek.com/gensim/models/word2vec.html).

#### bundes_w2v.py
Light wrapper to the Gensim w2v module.

#### data.py
Helper function to aquire the data.

#### utils.py 
Helper function to pre-process the data.

## Deployment
Other files are added to enable deployment of the API to Heroku and 
to have an automated workflow based on GitHub Actions.

Please note that you need to set environment variables to deploy on 
Google Cloud Platform. This needs to be done directly in `data.py`, 
`trainer.py` and `bundestrainer.py`. For the `MAKEFILE` environment
variables need to be set. This will replaced by a more flexible
approach in the future.

## Licence
[MIT](https://opensource.org/licenses/MIT)

## Team
The work is a colloborative effort of the following team members 
who each contributed to the project:

* [Anton Bauer](https://github.com/g-wagen)
* [Barbara Hartmann](https://github.com/BabsBerlin)
* [Felipe Lopes](https://github.com/felipebool)
* [Andreas Tai](https://github.com/xchange11)

## Thanks
We can not thank enough the AMAAAAAZIIIING team of Le Wagon.
The patience, expertise, and dedication opened a new world for us.