import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Charger le modèle de segmentation UNet
seg_model = tf.keras.models.load_model('C:/convert/streamlit/model_final3classes/model_unet.h5')

# Charger le modèle de classification DenseNet121
class_model = tf.keras.models.load_model('C:/convert/streamlit/model_final3classes/modelXceptionUvs.h5')

# Fonction pour segmenter les zones d'intérêt avec UNet
def segment_image(image):
    img = cv2.resize(image, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    mask = seg_model.predict(img)[0]
    mask = np.argmax(mask, axis=-1)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    return mask

# Fonction pour mettre en évidence les zones d'intérêt sur l'image
def highlight_regions(image, mask, color):
    overlay = image.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

# Interface Streamlit
st.title("Classification et segmentation d'images radiologiques")

uploaded_file = st.file_uploader("Choisir une image radiologique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prédire la classe de l'image avec DenseNet121
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 255.0
    class_pred = class_model.predict(img_resized)
    class_idx = np.argmax(class_pred)
    class_label = "Normal" if class_idx == 0 else "Pneumonie virale" if class_idx == 1 else "COVID-19"

    # Calculer les pourcentages
    class_prob = class_pred[0][class_idx] * 100
    covid_prob = class_pred[0][2] * 100 if class_pred[0][2] else 0
    pneumonia_prob = class_pred[0][1] * 100 if class_pred[0][1] else 0
    normal_prob = class_pred[0][0] * 100
    # Segmenter les zones d'intérêt avec UNet
    mask = segment_image(img)

    # Mettre en évidence les zones d'intérêt sur l'image
    highlighted_img = highlight_regions(img, mask, (255, 0, 0) if class_idx == 2 else (0, 0, 255) if class_idx == 1 else (0, 255, 0))

    # Afficher les résultats
    st.write(f"Classe prédite : {class_label}")
    st.write(f"Pourcentage de COVID-19 : {covid_prob:.2f}%")
    st.write(f"Pourcentage de pneumonie : {pneumonia_prob:.2f}%")
    st.write(f"Pourcentage de normalité : {normal_prob:.2f}%")
    st.image([img, highlighted_img], caption=["Image radiologique", "Zones d'intérêt"], width=300)

