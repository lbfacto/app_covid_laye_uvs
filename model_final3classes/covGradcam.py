import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Charger le modèle de segmentation UNet
seg_model = tf.keras.models.load_model('C:/convert/streamlit/model_final3classes/model_unet.h5')

# Charger le modèle de classification DenseNet121
class_model = tf.keras.models.load_model('C:/convert/streamlit/model_final3classes/modelXceptionUvs.h5')

# Fonction pour segmenter les zones d'intérêt avec UNet et les classer selon le modèle de classification
def segment_and_classify(image):
    # Prédiction de la segmentation
    pred_mask = seg_model.predict(image)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Application du masque à l'image
    covid_img = cv2.cvtColor(image[0], cv2.COLOR_RGB2GRAY)
    resized_mask = cv2.resize(pred_mask[0], (covid_img.shape[1], covid_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked_img = cv2.bitwise_and(covid_img, covid_img, mask=resized_mask)

    # Détection des contours de la zone infectée
    contours, _ = cv2.findContours(pred_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = cv2.cvtColor(covid_img, cv2.COLOR_GRAY2RGB) / 255.0
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
    # Calcul de la surface totale des deux poumons
    lung_mask = np.zeros_like(pred_mask[0])
    lung_mask[:, :256//2] = 1
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    lung_mask = cv2.resize(lung_mask, (covid_img.shape[1], covid_img.shape[0]))
    total_area = np.sum(lung_mask)

    # Calcul de la surface infectée
    infected_area = np.sum(pred_mask[0])

    # Calcul du pourcentage de zone infectée par rapport à la surface totale des deux poumons
    infected_percent = 100 * infected_area / total_area

    # Classer les zones d'intérêt selon le modèle de classification
    img_resized = cv2.resize(masked_img, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_resized = np.concatenate([img_resized] * 3, axis=-1)
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 255.0
    class_pred = class_model.predict(img_resized)
    class_idx = np.argmax(class_pred)
    class_label = "Normal" if class_idx == 0 else "Pneumonie virale" if class_idx == 1 else "COVID-19"

    # Encadrement des régions infectées en rouge
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return contour_img, infected_percent, class_label

# Interface utilisateur Streamlit
st.title("Segmentation d'images radiologiques et classification")
uploaded_file = st.file_uploader("Choisissez une image radiologique", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    img = image.load_img(uploaded_file, target_size=(256, 256))
    st.image(img, caption="Image radiologique", use_column_width=True)

    # Convertir l'image en tableau numpy
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Segmenter les zones d'intérêt et classer l'image
    segmented_img, infected_percent, class_label = segment_and_classify(img_array)

    # Afficher l'image segmentée avec les contours infectés et la classe prédite
    st.image(segmented_img, caption=f"Classe : {class_label}", use_column_width=True)
    st.write(f"Pourcentage de zone infectée : {infected_percent:.2f}%")
 