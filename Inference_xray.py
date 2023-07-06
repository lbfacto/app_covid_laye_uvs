from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
#resnet_chest = load_model('models/resnet_chest.h5')
#inception_chest = load_model('models/inceptionv3_chest.h5')
#xception_chest = load_model('models/xception_chest.h5')
model = load_model("C:/convert/streamlit/model_vgg_chest.h5", compile=False)

image = cv2.imread("C:/convert/streamlit/IM-0003-0001.jpeg") # read file
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
image = cv2.resize(image,(224,224))
image = np.array(image) / 255
image = np.expand_dims(image, axis=0)


pred = model.predict(image)
probability = pred[0]
print("vgg Predictions:")
if probability[0] > 0.5:

  pred_cov = str('%.2f' % (probability[0]*100) + '% COVID')
else:
  pred_cov = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
print(pred_cov)