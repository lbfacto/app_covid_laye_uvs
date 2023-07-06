
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC



c = 'E:/down_load/QuatarCov/covquatar1/COVID-19_Radiography_Dataset/COVID/images'
n = 'E:/down_load/QuatarCov/covquatar1/COVID-19_Radiography_Dataset/Normal/images'
p = 'E:/down_load/QuatarCov/covquatar1/COVID-19_Radiography_Dataset/Viral Pneumonia/images'


random.seed(42)
filenames = os.listdir(c) + random.sample(os.listdir(n), 2500) + os.listdir(p)


categories = []
for filename in filenames:
    category = filename.split('-')[0]
    if category == 'COVID':
        categories.append(str(2))
    elif category == 'Viral Pneumonia':
        categories.append(str(1))
    else:
        categories.append(str(0))
for i in range(len(filenames)):
    if 'COVID' in filenames[i]:
        filenames[i] = os.path.join(c, filenames[i])
    elif 'Viral Pneumonia' in filenames[i]:
        filenames[i] = os.path.join(p, filenames[i])
    else:
        filenames[i] = os.path.join(n, filenames[i])
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})




train_data, test_valid_data = train_test_split(df, test_size=0.2, random_state = 42, shuffle=True, stratify=df['category'])
train_data = train_data.reset_index(drop=True)
test_valid_data = test_valid_data.reset_index(drop=True)
test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, random_state = 42,
                                        shuffle=True, stratify=test_valid_data['category'])
test_data = test_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)


train_data_gen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_data_gen.flow_from_dataframe(
    train_data,
    x_col='filename',
    y_col='category',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=256,
    shuffle=False
)
valid_data_gen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_data_gen.flow_from_dataframe(
    valid_data,
    x_col='filename',
    y_col='category',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=256,
    shuffle=False
    )
test_set = valid_data_gen.flow_from_dataframe(
    test_data,
    x_col='filename',
    y_col='category',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=256,
    shuffle=False
)

METRICS = [TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    'accuracy',
    Precision(name='precision'),
    Recall(name='recall')]

optimizer= ["adam", "rmsprop"]
loss= ["binary_crossentropy", "categorical_crossentropy"]
metrics= METRICS



#mlflow.tensorflow.autolog(registered_model_name=f"Model_cnn01_transfer_Learning{run_date}")
#if mlflow.get_experiment_by_name(f"run_{run_date}") == None:
    #mlflow.create_experiment(f"run_{run_date}")
    #mlflow.set_experiment(f"run_{run_date}")
    #mlflow.set_experiment("fine_tuning_model")

#mlflow.set_experiment(f"run_{run_date}")
#with mlflow.start_run(run_name="Model_cnn01_transfer_Learning") as run:
#model=tf.keras.models.load_model("C:\convert\streamlit\model_final3classes\modelXceptionUvs.h5", compile=True)
    #mlflow.log_param("optimizer", optimizer)
    #mlflow.log_param("loss", loss)
    #mlflow.log_param("metric", METRICS)
