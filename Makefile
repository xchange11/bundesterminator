# ----------------------------------
#        LOOK FOR .env FILE
# ----------------------------------
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* bundestag/*.py

black:
	@black scripts/* bundestag/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr bundestag-*.dist-info
	@rm -fr bundestag.egg-info

install:
	@pip install . -U

all: clean install test black check_code

run_api:
	uvicorn api.bundestag:app --reload

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      GCLOUD INTEGRATION
# ----------------------------------
PACKAGE_NAME=bundestag
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

SPEECH_SEGMENTS_A_PATH=raw_data/speech_segments_a.csv
SPEECH_SEGMENTS_B_PATH=raw_data/speech_segments_b.csv
BIO_DATA_PATH=raw_data/bio_data.csv

BUCKET_DATA_FOLDER=trained
SPEECH_A_BUCKET_FILE_NAME=$(shell basename ${SPEECH_SEGMENTS_A_PATH})
SPEECH_B_BUCKET_FILE_NAME=$(shell basename ${SPEECH_SEGMENTS_B_PATH})
BIO_BUCKET_FILE_NAME=$(shell basename ${BIO_DATA_PATH})
BUCKET_TRAINING_FOLDER=trainings
FILENAME_FIRST=trainer
FILENAME_SECOND=bundestrainer

JOB_NAME=bundestag_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

set_project:
	@gcloud config set project ${GCP_PROJECT_ID}

create_bucket:
	@gsutil mb -l ${GCP_REGION} -p ${GCP_PROJECT_ID} gs://${GCP_BUCKET_NAME}

upload_data:
	@gsutil cp ${SPEECH_SEGMENTS_A_PATH} gs://${GCP_BUCKET_NAME}/${BUCKET_DATA_FOLDER}/${SPEECH_A_BUCKET_FILE_NAME}
	@gsutil cp ${SPEECH_SEGMENTS_B_PATH} gs://${GCP_BUCKET_NAME}/${BUCKET_DATA_FOLDER}/${SPEECH_B_BUCKET_FILE_NAME}
	@gsutil cp ${BIO_DATA_PATH} gs://${GCP_BUCKET_NAME}/${BUCKET_DATA_FOLDER}/${BIO_BUCKET_FILE_NAME}

gcp_submit_first_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${GCP_BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_FIRST} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${GCP_REGION} \
		--stream-logs

gcp_submit_second_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${GCP_BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_SECOND} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${GCP_REGION} \
		--stream-logs

update_first_model:
	@gsutil cp gs://${GCP_BUCKET_NAME}/model.joblib api/model.joblib

update_second_model:
	@gsutil cp gs://${GCP_BUCKET_NAME}/model2.tf api/model2.tf

update_w2v_model:
	@gsutil cp gs://${GCP_BUCKET_NAME}/model2.w2v api/model2.w2v

update_party_mapping:
	@gsutil cp gs://${GCP_BUCKET_NAME}/model2.pm api/model2.pm

do_gcp_setup: set_project create_bucket

do_gcp_first_model_training: upload_data gcp_submit_first_training

do_gcp_second_model_training: upload_data gcp_submit_second_training
