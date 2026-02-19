import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI Farb-Battle: Pentagon Edition", page_icon="🎨")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return model, class_names

def draw_pentagon(active_color=None):
    # Die 5 Ecken des Pentagons
    categories = ['Rot', 'Gelb', 'Lila', 'Grün', 'Blau']
    # Pentagon-Regeln Logik für die Anzeige (Stärke-Punkte)
    rules = {
        'Rot': ['Grün', 'Gelb'],
        'Gelb': ['Blau', 'Lila'],
        'Blau': ['Grün', 'Rot'],
        'Grün': ['Gelb', 'Lila'],
        'Lila': ['Rot', 'Blau']
    }
    
    # Werte für das Diagramm (1 wenn aktiv, sonst 0.2 für die Form)
    values = [1 if c == active_color else 0.3 for c in categories]
    values.append(values[0]) # Kreis schließen
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    
    # Das Grund-Pentagon
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        name='Deine Wahl',
        line=dict(color='#FF4B4B' if active_color else '#BDC3C7')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=14), rotation=90, direction="clockwise")
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300
    )
    return fig

def predict_color(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

def determine_winner(player, computer):
    rules = {'Rot': ['Grün', 'Gelb'], 'Gelb': ['Blau', 'Lila'], 'Blau': ['Grün', 'Rot'], 'Grün': ['Gelb', 'Lila'], 'Lila': ['Rot', 'Blau']}
    if player == computer: return "unentschieden"
    return "player" if computer in rules.get(player, []) else "computer"

# --- SESSION STATE ---
if 'player_score' not in st.session_state:
    st.session_state.update({'player_score': 0, 'computer_score': 0, 'game_over': False})

# --- UI ---
st.title("🎨 KI Farb-Battle: Best of 7")

# Pentagon Info-Bereich
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader(f"Spielstand: {st.session_state.player_score} : {st.session_state.computer_score}")
    with st.expander("📖 Kampf-Regeln"):
        st.write("🔴 > 🟢,🟡 | 🟡 > 🔵,🟣")
        st.write("🔵 > 🟢,🔴 | 🟢 > 🟡,🟣")
        st.write("🟣 > 🔴,🔵")
with col_b:
    # Zeige ein leeres Pentagon als Guide
    st.plotly_chart(draw_pentagon(), use_container_width=True, config={'displayModeBar': False})

st.divider()

if not st.session_state.game_over:
    img_file = st.camera_input("Scanner") or st.file_uploader("Upload")

    if img_file:
        model, class_names = load_model_and_labels()
        image = Image.open(img_file).convert("RGB")
        color_detected, confidence = predict_color(image, model, class_names)
        
        # KI Zug
        comp_choice = np.random.choice(['Rot', 'Gelb', 'Grün', 'Blau', 'Lila'])
        
        # Visualisierung des Kampfes
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Deine Farbe", color_detected, f"{int(confidence*100)}% sicher")
            st.plotly_chart(draw_pentagon(color_detected), use_container_width=True, config={'displayModeBar': False})
        
        with c2:
            st.metric("KI Wahl", comp_choice)
            st.title({"Rot":"🔴","Gelb":"🟡","Grün":"🟢","Blau":"🔵","Lila":"🟣"}.get(comp_choice))

        # Ergebnis
        result = determine_winner(color_detected, comp_choice)
        if result == "player":
            st.success(f"Sieg! {color_detected} schlägt {comp_choice}"); st.session_state.player_score += 1
        elif result == "computer":
            st.error(f"Niederlage! {comp_choice} schlägt {color_detected}"); st.session_state.computer_score += 1
        else:
            st.info("Unentschieden!")

        if st.session_state.player_score >= 4 or st.session_state.computer_score >= 4:
            st.session_state.game_over = True
            st.rerun()
else:
    if st.session_state.player_score >= 4: st.success("# 🎉 MATCH-SIEG!"); st.balloons()
    else: st.error("# 🤖 KI GEWINNT DAS MATCH!")
    if st.button("Revanche?"):
        st.session_state.update({'player_score': 0, 'computer_score': 0, 'game_over': False})
        st.rerun()
