import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import io
import cv2
import webbrowser
import streamlit as st
import streamlit.components.v1 as components




#titre de l
model = tf.keras.models.load_model('C:/convert/streamlit/vgg_ct.h5', compile=False)
#titre de l
def predict(image):
    # Prétraitement de l'image
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prédiction avec le modèle
    prediction = model.predict(img_array)

    return prediction[0][0]

# Interface utilisateur Streamlit
st.title("Prédiction COVID-19")
uploaded_file = st.file_uploader("Téléchargez une image au format X-ray", type=["jpg","jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image téléchargée
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Prédiction avec le modèle
    prediction = predict(image)
    if prediction > 0.5:
        st.write("Prédiction : COVID-19")
    else:
        st.write("Prédiction : Non COVID-19")