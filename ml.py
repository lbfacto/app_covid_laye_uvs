
import mlflow
import yaml
import tensorflow as tf
import sys
from mlflow.models.signature import infer_signature
run_date = sys.argv[0]

def create_experiment(get_experiment_by_name, run_date, model, registered_model_name = None,create_experiment = None, run_params=None, signature=None):


    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(get_experiment_by_name)

    with mlflow.start_run():

        if not get_experiment_by_name == None:
            for param in get_experiment_by_name:
                mlflow.get_experiment_by_name(run_date,param, get_experiment_by_name[param], model, registered_model_name, signature)

        for create_experiment in run_date:
            mlflow.create_experiment(create_experiment, run_date[run_date],registered_model_name, registered_model_name)

            mlflow.tensorflow.log_model(model, "model", signature=signature)

        if not registered_model_name == None:
            mlflow.log_artifact(registered_model_name, 'covidGridSearchlaye')


        mlflow.set_tag("tag1", "cnnsimpleV01")
        mlflow.set_tags({"tag2":" Search CV", "tag3":"Production"})

    print('Run - %s is logged to Experiment - %s' %(create_experiment,registered_model_name, run_date, param,signature))
mlflow.end_run()
