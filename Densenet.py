from keras.models import Sequential, load_model
from packages import*
from tensorflow.keras.initializers import GlorotNormal
densenet = keras.applications.densenet
conv_model = densenet.DenseNet121(weights='imagenet', include_top=False, pooling="avg", input_shape=(224,224,3))
x = keras.layers.Dropout(0.3)(conv_model.output)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='tanh',kernel_initializer=GlorotNormal(),bias_regularizer=tf.keras.regularizers.L2(0.0001), kernel_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer = tf.keras.regularizers.L2(0.0001))(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(32, activation='tanh',kernel_initializer=GlorotNormal(),bias_regularizer=tf.keras.regularizers.L2(0.0001) ,kernel_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer = tf.keras.regularizers.L2(0.0001))(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(16, activation='tanh',kernel_initializer=GlorotNormal(),bias_regularizer=tf.keras.regularizers.L2(0.0001) ,kernel_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer = tf.keras.regularizers.L2(0.0001))(x)
x = keras.layers.Dropout(0.3)(x)
predictions = keras.layers.Dense(3, activation='softmax', kernel_initializer=GlorotNormal(), bias_regularizer=tf.keras.regularizers.L2(0.0001),kernel_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer = tf.keras.regularizers.L2(0.0001))(x)
model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
for layer in model.layers[:300]:
#for layer in model.layers:

      layer.trainable = False
for layer in model.layers[300:]:
#for layer in model.layers:

      layer.trainable = True
  # https://www.kaggle.com/datasets/theewok/chexnet-keras-weights
  #densenet_model.load_weights("/content/drive/MyDrive/covquatar1/brucechou1983_CheXNet_Keras_0.3.0_weights (2).h5", by_name = True, skip_mismatch = True)
