import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import pandas as pd
import io
import cv2
import webbrowser
import streamlit_menu as men
import streamlit as st
import streamlit.components.v1 as components
import base64
import io
from io import BytesIO
import base64
import mlflow
import sys


run_date = sys.argv[0]


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("C:/convert/streamlit/img1.png")

page_bg_img = f"""
<style>

[data-testid ="stAppViewContainer"] > .main {{
background-image:url("https://www.unchk.sn/wp-content/uploads/2023/03/eno1.jpg");
background-size :250%;

background-position: midle;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");

background-position: left;
background-repeat: no-repeat;
background-attachment: fixed;
    text-align: left;
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 110%;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 3rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("MENU")


# Ajout du contenu de votre application Streamlit
# st.write("Mon application Streamlit")


url = 'https://www.gmail.com/'
logo = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')
with st.sidebar:
    choose = option_menu("IA de Segmentation et de Classification du COVID-19", ["A propos", "Classification", "Prediction covid-19 xray", "Segmentation_semantic"],
                         icons=['house', 'bi bi-graph-down-arrow',
                                'bi bi-graph-down-arrow', 'house', 'house'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "15!important", "background-color": "#b36b00"},
        "icon": {"color": "blue", "font-size": "25px"},
        "nav-link": {"font-size": "10px", "text-align": "left", "margin": "0px", "--hover-color": "orange"},
        "nav-link-selected": {"background-color": "#4FC031"},
    }
    )

profile = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')
url = 'https://www.gmail.com/'
if (choose == "A propos"):
    col1, col2 = st.columns([0.8, 0.2])

    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
    font-size: 25px ; font-family: 'Cooper green'; color: green}
    </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">A propos du createur</p>',
                    unsafe_allow_html=True)
    with col2:               # To display brand log
        st.image(logo, width=10)

    st.write("Abdoulaye BA MASTER 2 BIG DATA ANALYTICS Univetsite Numerique cheikh Amadou KANE de Dakar. Memoire de deep learning applique sur des images thoraciques de covid-19 le lien du repos sur github est disponibles sur ce lien: https://github.com/lbfacto metric sur tensorboard dev des accurance https://tensorboard.dev/experiment/dy6HW7GURH2pSXBpzwtx1w/#histograms&run=20230506-032808%2Ftrain")
    st.image(profile, width=100)

elif (choose == "Classification"):
    # st.title("Predictioon du covid-19 sur des images tomographioque des poumons")
    # Start mlflow
    # mlflow.set_tracking_uri(f"http://127.0.0.1:5000")
    seg_model = tf.keras.models.load_model(
        'C:/convert/streamlit/model_final3classes/model_unet.h5', compile=True)

    # Charger le modèle de classification DenseNet121
    class_model = tf.keras.models.load_model(
        'C:\streamlit\model\model_uvs_1.h5')


    @st.cache_data
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
    @st.cache_data
    def highlight_regions(image, mask, color):
        overlay = image.copy()
        overlay[mask == 1] = color
        return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # Interface Streamlit
    st.title("Classification et segmentation d'images radiologiques")

    uploaded_file = st.file_uploader(
        "Choisir une image radiologique", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Charger l'image
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
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
        highlighted_img = highlight_regions(img, mask, (255, 0, 0) if class_idx == 2 else (
            0, 0, 255) if class_idx == 1 else (0, 255, 0))

        # Afficher les résultats
        st.write(f"Classe prédite : {class_label}")
        st.write(f"Pourcentage de COVID-19 : {covid_prob:.2f}%")
        st.write(f"Pourcentage de pneumonie : {pneumonia_prob:.2f}%")
        st.write(f"Pourcentage de normalité : {normal_prob:.2f}%")
        st.image([img, highlighted_img], caption=[
                 "Image radiologique", "Zones d'intérêt"], width=300)

    # titre de l

elif (choose == "Prediction covid-19 xray"):
    st.title("Predictioon du covid-19 sur des images radiographiques du thorax(XRAY)")
# titre de l

    model = tf.keras.models.load_model(
        'C:\convert\streamlit/vgg_chest.h5', compile=False)


# Définition de la fonction de prédiction

    @st.cache_data
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
    # st.title("Prédiction COVID-19")
    uploaded_file = st.file_uploader(
        "Téléchargez une image radiographique du thorax ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Affichage de l'image téléchargée
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Image téléchargé', use_column_width=True)
        # Prédiction avec le modèle
        # st.image(image, caption='Image de radiographie pulmonaire', use_column_width=True)
        proba = predict(image) * 100.0
        if proba > 50:
            st.error(
                "COVID-19 détecté avec une probabilité de {:.2f}%".format(proba))
        else:
            st.success(
                "Pas de COVID-19 détecté avec une probabilité de {:.2f}%".format(100-proba))

elif (choose == "Prediction-ctglobal"):
    st.title(
        "Predictioon Covid-19 sur les images du thorax comme sur des images tomographique -")
    model = tf.keras.models.load_model(
        'C:/convert/streamlit/vgg_ct.h5', compile=False)
# titre de l

    @st.cache_data
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
    # st.title("Prédiction COVID-19")
    uploaded_file = st.file_uploader(
        "Téléchargez une image pour la prediction", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Affichage de l'image téléchargée
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        proba = predict(image) * 100.0
        if proba > 50:
            st.error(
                "COVID-19 détecté avec une probabilité de {:.2f}%".format(proba))
        else:
            st.success(
                "Pas de COVID-19 détecté avec une probabilité de {:.2f}%".format(100-proba))
        # Prédiction avec le modèle
        # prediction = predict(image)
        # if prediction > 0.5:
        # st.write("Prédiction : COVID-19")
        # else:
        # st.write("Prédiction : Non COVID-19")
elif (choose == "Segmentation_semantic"):
    st.title("Segmenatation d'image Thoraciques et détection des zone d'ineteret COVID-19 sur des radiographies pulmonaires")

    model = tf.keras.models.load_model(
        'C:/convert/streamlit/model_unet.h5', compile=False)

    # Fonction de prédiction

    def predict(image):
        # Chargement de l'image
        covid_img = np.array(image.convert('RGB'))

        # Prétraitement de l'image pour la segmentation
        test_img = cv2.resize(covid_img, (256, 256))
        test_img = np.expand_dims(test_img, axis=0)
        test_img = test_img.astype('float32') / 255

        # Prédiction de la segmentation
        pred_mask = model.predict(test_img)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Application du masque à l'image
        covid_img = cv2.cvtColor(covid_img, cv2.COLOR_RGB2GRAY)
        # redimensionner le masque
        resized_mask = cv2.resize(
            pred_mask[0], (covid_img.shape[1], covid_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # appliquer le masque à l'image n
        # masked_img = cv2.bitwise_and(covid_img, covid_img, masked_img=resized_mask)

        # Détection des contours de la zone infectée
        contours, _ = cv2.findContours(
            pred_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = cv2.cvtColor(covid_img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_img, contours, -1, (255, 10, 0))  # , 3)

        # Calcul de la surface totale des deux poumons
        lung_mask = np.zeros_like(pred_mask[0])
        lung_mask[:, :256//2] = 1
        lung_mask = cv2.morphologyEx(
            lung_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        lung_mask = cv2.resize(
            lung_mask, (covid_img.shape[1], covid_img.shape[0]))
        total_area = np.sum(lung_mask)

        # Calcul de la surface infectée
        infected_area = np.sum(pred_mask[0])

        # Calcul du pourcentage de zone infectée par rapport à la surface totale des deux poumons
        infected_percent = 100 * infected_area / total_area

        # Encadrement des régions infectées en rouge
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_img, (x, y), (x+w, y+h),
                          (255, 100, 200))  # , 2)

        # Affichage de l'image résultat avec le pourcentage de zone infectée
        st.image(contour_img, use_column_width=True)
        st.write(f"Pourcentage de zone infectée : {infected_percent:.2f}%")

    # Interface utilisateur Streamlit

    uploaded_file = st.file_uploader(
        "Choisir une image de radiographie pulmonaire", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption='Image d\'entrée', use_column_width=True)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        predict_button = st.button("Segmenter", key="segment_button")

        if predict_button:
            # Effectuer la prédiction et afficher les résultats
            predict(image)

    # if predict_button:
    # predict(image)


# titre de l
