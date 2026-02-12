import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI Farb-Battle: Best of 7", page_icon="🎨")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_and_labels():
    # Lädt das Modell und die Labels
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return model, class_names

def predict_color(image, model, class_names):
    # Bildvorbereitung (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalisierung [-1, 1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

def get_computer_choice():
    # Die KI wählt zufällig eine Farbe aus der Liste
    return np.random.choice(['Rot', 'Gelb', 'Grün', 'Blau', 'Lila'])

def determine_winner(player, computer):
    # Die Pentagon-Logik:
    # Rot > (Grün, Gelb); Gelb > (Blau, Lila); Blau > (Grün, Rot); Grün > (Gelb, Lila); Lila > (Rot, Blau)
    rules = {
        'Rot': ['Grün', 'Gelb'],
        'Gelb': ['Blau', 'Lila'],
        'Blau': ['Grün', 'Rot'],
        'Grün': ['Gelb', 'Lila'],
        'Lila': ['Rot', 'Blau']
    }
    
    if player == computer:
        return "unentschieden"
    if computer in rules[player]:
        return "player"
    else:
        return "computer"

# --- INITIALISIERUNG SESSION STATE ---
if 'player_score' not in st.session_state:
    st.session_state.player_score = 0
if 'computer_score' not in st.session_state:
    st.session_state.computer_score = 0
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

def reset_game():
    st.session_state.player_score = 0
    st.session_state.computer_score = 0
    st.session_state.game_over = False

# --- UI LAYOUT ---
st.title("🎨 KI Farb-Battle: Best of 7")

with st.expander("ℹ️ Spielregeln & Logik (Pentagon-System)"):
    st.write("""
    Schlage die KI in einem Best-of-7 Match! 
    - 🔴 **Rot** schlägt 🟢 Grün & 🟡 Gelb
    - 🟡 **Gelb** schlägt 🔵 Blau & 🟣 Lila
    - 🔵 **Blau** schlägt 🟢 Grün & 🔴 Rot
    - 🟢 **Grün** schlägt 🟡 Gelb & 🟣 Lila
    - 🟣 **Lila** schlägt 🔴 Rot & 🔵 Blau
    """)

# Punktestand Anzeige
st.subheader(f"Spielstand: Spieler {st.session_state.player_score} : {st.session_state.computer_score} Computer")

if not st.session_state.game_over:
    # Eingabemethoden
    tab1, tab2 = st.tabs(["📷 Kamera nutzen", "📁 Bild hochladen"])
    
    img_file = None
    with tab1:
        img_file = st.camera_input("Zeige der KI deine Farbe!")
    with tab2:
        img_upload = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "png", "jpeg"])
        if img_upload:
            img_file = img_upload

    if img_file:
        model, class_names = load_model_and_labels()
        image = Image.open(img_file).convert("RGB")
        
        # Erkennung
        color_detected, confidence = predict_color(image, model, class_names)
        confidence_pct = round(confidence * 100)
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### Deine Farbe: **{color_detected}**")
            st.write(f"Sicherheit: {confidence_pct}%")
            if confidence < 0.7:
                st.warning(f"Ich bin mir nicht ganz sicher, ob das {color_detected} ist...")
        
        # KI-Zug
        comp_choice = get_computer_choice()
        with col2:
            st.write(f"### KI wählt: **{comp_choice}**")
            emoji_map = {"Rot": "🔴", "Gelb": "🟡", "Grün": "🟢", "Blau": "🔵", "Lila": "🟣"}
            st.title(emoji_map.get(comp_choice, "❓"))

        # Ergebnis berechnen
        result = determine_winner(color_detected, comp_choice)
        
        if result == "player":
            st.success(f"Sieg! {color_detected} schlägt {comp_choice}!")
            st.session_state.player_score += 1
        elif result == "computer":
            st.error(f"Niederlage! {comp_choice} schlägt {color_detected}!")
            st.session_state.computer_score += 1
        else:
            st.info("Unentschieden! Gleiche Farbe gewählt.")

        # Siegprüfung
        if st.session_state.player_score >= 4 or st.session_state.computer_score >= 4:
            st.session_state.game_over = True
            st.rerun()

else:
    # Großes Finale
    if st.session_state.player_score >= 4:
        st.balloons()
        st.success("# 🎉 GRATULATION! Du hast das Match gewonnen!")
    else:
        st.error("# 🤖 Die KI gewinnt das Match! Viel Erfolg beim nächsten Mal.")
    
    if st.button("Match neustarten"):
        reset_game()
        st.rerun()
