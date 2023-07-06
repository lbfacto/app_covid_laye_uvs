
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.model_selection import KFold
from sklearn.model_selection import  ParameterGrid

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from porcessing import*
from ml import*
from packages import*
from Densenet import*
#from porcessing import*

METRICS =  [TruePositives(name='tp'),
                                                        FalsePositives(name='fp'),
                                                        TrueNegatives(name='tn'),
                                                        FalseNegatives(name='fn'),
                                                        'accuracy',
                                                        Precision(name='precision'),
                                                        Recall(name='recall')]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3, mode = 'min')
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                            patience=2,
                                                            verbose=2,
                                                            factor=0.5,
                                                            min_lr=0.00001)
reduce_lr =  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=3, min_lr=0.00001)

def get_model_name(k):
    return 'model_uvs_'+str(k)+'.h5'

VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []

save_dir = 'C:/convert/streamlit/model/'
fold_var = 1


def create_model():
# create model
    model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
# Compile model

    return model
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var),
                            monitor='val_accuracy', verbose=1,
                            save_best_only=True, mode='max')
callbacks_list = [checkpoint]
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL

param_grid = {
    "optimizer": ["adam", "rmsprop"],
    "loss": ["categorical_crossentropy"],
    "metrics": METRICS
}
#kfold = KFold(n_splits=20, shuffle=True)

kf = KFold(n_splits = 10)

skf = StratifiedKFold(n_splits = 10, random_state = 7, shuffle = True)




for optimizer in param_grid["optimizer"]:
    for loss in param_grid["loss"]:
        for metric in param_grid["metrics"]:
            mlflow.tensorflow.autolog(registered_model_name="covidlaye")
            if mlflow.get_experiment_by_name("run_date") == None:
                    mlflow.create_experiment("covidlaye")
                    mlflow.tensorflow.log_model(model, "model")
                    #mlflow.set_experiment("run_date")
            mlflow.set_experiment("covidlaye")
            with mlflow.start_run(run_name='covidModellaye'):
                model.compile(optimizer=optimizer,loss = 'categorical_crossentropy',metrics = METRICS)

        #model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
                hist = model.fit(train_generator, epochs=10,validation_data=valid_generator,
                    callbacks=[callbacks_list, reduce_lr, learning_rate_reduction,early_stopping])

                mlflow.log_param("optimizer", optimizer)
                mlflow.log_param("loss", loss)
                mlflow.log_param("metric", metric)
                mlflow.log_metric("accuracy", hist.history['accuracy'][-1])
                mlflow.log_metric("val_accuracy", hist.history['val_accuracy'][-1])

                model.save('covid.h5')


model.load_weights("C:/convert/streamlit/model/model_uvs_"+str(fold_var)+".h5")


results = model.evaluate(test_set)
results = dict(zip(model.metrics_names,results))

VALIDATION_ACCURACY.append(results['accuracy'])
VALIDAITON_LOSS.append(results['loss'])

tf.keras.backend.clear_session()

fold_var += 1
mlflow.end_run()