import streamlit as st

# --- WICHTIGER FIX FÜR KERAS 3 / PYTHON 3.13 ---
# Dieser Teil muss VOR dem Import von TensorFlow stehen!
import keras
from keras import layers

# Wir bringen Keras bei, das veraltete 'groups'-Argument zu ignorieren
original_depthwise_init = layers.DepthwiseConv2D.__init__
def patched_depthwise_init(self, *args, **kwargs):
    if "groups" in kwargs:
        kwargs.pop("groups")
    return original_depthwise_init(self, *args, **kwargs)
layers.DepthwiseConv2D.__init__ = patched_depthwise_init
# ----------------------------------------------

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI Farb-Battle: Best of 7", page_icon="🎨")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_and_labels():
    # Lädt das Modell ohne Kompilierung (verhindert Optimizer-Fehler)
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding='utf-8') as f:
        # Bereinigt die Labels (entfernt Zahlen wie '0 Rot' -> 'Rot')
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return model, class_names

def predict_color(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

def get_computer_choice():
    return np.random.choice(['Rot', 'Gelb', 'Grün', 'Blau', 'Lila'])

def determine_winner(player, computer):
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
    st.session_state.update({'player_score': 0, 'computer_score': 0, 'game_over': False})

# --- UI ---
st.title("🎨 KI Farb-Battle: Best of 7")

with st.expander("ℹ️ Spielregeln (Pentagon-Logik)"):
    st.write("🔴 > 🟢,🟡 | 🟡 > 🔵,🟣 | 🔵 > 🟢,🔴 | 🟢 > 🟡,🟣 | 🟣 > 🔴,🔵")

st.subheader(f"Spielstand: Spieler {st.session_state.player_score} : {st.session_state.computer_score} Computer")

if not st.session_state.game_over:
    img_file = st.camera_input("Zeige deine Farbe!") or st.file_uploader("Oder Bild hochladen")

    if img_file:
        model, class_names = load_model_and_labels()
        image = Image.open(img_file).convert("RGB")
        color_detected, confidence = predict_color(image, model, class_names)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### Deine Farbe: **{color_detected}**")
            st.write(f"Sicherheit: {int(confidence*100)}%")
            if confidence < 0.7: st.warning("Unsicher...")

        comp_choice = get_computer_choice()
        with col2:
            st.write(f"### KI wählt: **{comp_choice}**")
            st.title({"Rot":"🔴","Gelb":"🟡","Grün":"🟢","Blau":"🔵","Lila":"🟣"}.get(comp_choice))

        result = determine_winner(color_detected, comp_choice)
        if result == "player":
            st.success("Punkt für dich!"); st.session_state.player_score += 1
        elif result == "computer":
            st.error("Punkt für KI!"); st.session_state.computer_score += 1
        else:
            st.info("Unentschieden!")

        if st.session_state.player_score >= 4 or st.session_state.computer_score >= 4:
            st.session_state.game_over = True
            st.rerun()
else:
    if st.session_state.player_score >= 4: st.success("# 🎉 SIEG!"); st.balloons()
    else: st.error("# 🤖 KI GEWINNT!")
    if st.button("Neustart"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()
