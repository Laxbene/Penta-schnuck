import streamlit as st

# --- WICHTIGER FIX FÜR KERAS 3 / PYTHON 3.13 ---
import os
# Wir erzwingen, dass Keras das alte Format nutzt
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

# 1. Fix für das 'groups' Problem
original_depthwise_init = layers.DepthwiseConv2D.__init__
def patched_depthwise_init(self, *args, **kwargs):
    if "groups" in kwargs:
        kwargs.pop("groups")
    return original_depthwise_init(self, *args, **kwargs)
layers.DepthwiseConv2D.__init__ = patched_depthwise_init

# 2. Fix für das 'ValueError' / Input_Spec Problem
# Wir erlauben dem Modell, die Eingangsgröße flexibler zu handhaben
original_sequential_add = keras.models.Sequential.add
def patched_sequential_add(self, layer):
    if hasattr(layer, '_batch_input_shape'):
        # Falls das Modell hakt, setzen wir die Spec manuell auf None
        layer.input_spec = None 
    return original_sequential_add(self, layer)
keras.models.Sequential.add = patched_sequential_add
# ----------------------------------------------

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI Farb-Battle: Best of 7", page_icon="🎨")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_and_labels():
    # Wir laden das Modell mit custom_objects, falls nötig, 
    # aber compile=False ist der wichtigste Teil
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    
    with open('labels.txt', 'r', encoding='utf-8') as f:
        # Bereinigt '0 Rot' zu 'Rot'
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return model, class_names

def predict_color(image, model, class_names):
    # Bildvorbereitung exakt wie von Teachable Machine gefordert
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalisierung: (Pixel / 127.5) - 1 -> ergibt Bereich [-1, 1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

def get_computer_choice():
    return np.random.choice(['Rot', 'Gelb', 'Grün', 'Blau', 'Lila'])

def determine_winner(player, computer):
    # Pentagon-Regeln
    rules = {
        'Rot': ['Grün', 'Gelb'],
        'Gelb': ['Blau', 'Lila'],
        'Blau': ['Grün', 'Rot'],
        'Grün': ['Gelb', 'Lila'],
        'Lila': ['Rot', 'Blau']
    }
    if player == computer: return "unentschieden"
    return "player" if computer in rules.get(player, []) else "computer"

# --- SESSION STATE ---
if 'player_score' not in st.session_state:
    st.session_state.player_score = 0
    st.session_state.computer_score = 0
    st.session_state.game_over = False

# --- UI ---
st.title("🎨 KI Farb-Battle: Best of 7")

with st.expander("ℹ️ Spielregeln & Logik"):
    st.write("Schlage die KI! Das Pentagon-System bestimmt den Sieger:")
    st.write("🔴 > 🟢,🟡 | 🟡 > 🔵,🟣 | 🔵 > 🟢,🔴 | 🟢 > 🟡,🟣 | 🟣 > 🔴,🔵")

st.subheader(f"Spielstand: Spieler {st.session_state.player_score} : {st.session_state.computer_score} Computer")

if not st.session_state.game_over:
    # Kamera und Upload
    img_file = st.camera_input("Zeige deine Farbe!")
    if not img_file:
        img_file = st.file_uploader("Oder Bild hochladen", type=["jpg", "png", "jpeg"])

    if img_file:
        model, class_names = load_model_and_labels()
        image = Image.open(img_file).convert("RGB")
        
        # Vorhersage
        color_detected, confidence = predict_color(image, model, class_names)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### Deine Farbe: **{color_detected}**")
            st.write(f"Sicherheit: {int(confidence*100)}%")
            if confidence < 0.7:
                st.warning(f"Ich bin mir nicht sicher, ob das {color_detected} ist...")

        # KI Zug
        comp_choice = get_computer_choice()
        with col2:
            st.write(f"### KI wählt: **{comp_choice}**")
            emojis = {"Rot":"🔴","Gelb":"🟡","Grün":"🟢","Blau":"🔵","Lila":"🟣"}
            st.title(emojis.get(comp_choice, "❓"))

        # Ergebnis
        result = determine_winner(color_detected, comp_choice)
        if result == "player":
            st.success(f"Punkt für dich! {color_detected} gewinnt.")
            st.session_state.player_score += 1
        elif result == "computer":
            st.error(f"Punkt für die KI! {comp_choice} gewinnt.")
            st.session_state.computer_score += 1
        else:
            st.info("Unentschieden! Nochmal.")

        # Match-Ende prüfen
        if st.session_state.player_score >= 4 or st.session_state.computer_score >= 4:
            st.session_state.game_over = True
            st.rerun()

else:
    # Siegesmeldung
    if st.session_state.player_score >= 4:
        st.balloons()
        st.success("# 🎉 GRATULATION! Du hast das Match gewonnen!")
    else:
        st.error("# 🤖 Die KI gewinnt das Match!")
    
    if st.button("Neues Match starten"):
        st.session_state.player_score = 0
        st.session_state.computer_score = 0
        st.session_state.game_over = False
        st.rerun()
