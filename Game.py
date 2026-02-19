import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI Farb-Battle: Pentagon", page_icon="🎨", layout="centered")

# --- MODELL-LOGIK ---
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return model, class_names

# --- PENTAGON VISUALISIERUNG ---
def draw_interactive_pentagon(player_choice=None, comp_choice=None):
    # Festgelegte Reihenfolge für das Fünfeck
    nodes = ['Rot', 'Gelb', 'Lila', 'Blau', 'Grün']
    colors = {'Rot': 'red', 'Gelb': 'gold', 'Lila': 'purple', 'Blau': 'blue', 'Grün': 'green'}
    
    # Koordinaten berechnen (Einheitskreis)
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False).tolist()
    # Rotation, damit Rot oben ist
    angles = [a + np.pi/2 for a in angles]
    
    x = [np.cos(a) for a in angles]
    y = [np.sin(a) for a in angles]
    
    fig = go.Figure()

    # Gewinn-Logik Linien (Wer schlägt wen)
    rules = {
        'Rot': ['Grün', 'Gelb'], 'Gelb': ['Blau', 'Lila'], 'Blau': ['Grün', 'Rot'],
        'Grün': ['Gelb', 'Lila'], 'Lila': ['Rot', 'Blau']
    }

    # Pfeile für die Regeln zeichnen
    for start_node, targets in rules.items():
        start_idx = nodes.index(start_node)
        for target_node in targets:
            target_idx = nodes.index(target_node)
            fig.add_annotation(
                x=x[target_idx], y=y[target_idx],
                ax=x[start_idx], ay=y[start_idx],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor="rgba(150, 150, 150, 0.4)"
            )

    # Punkte für die Farben zeichnen
    for i, name in enumerate(nodes):
        size = 30
        color = colors[name]
        opacity = 1.0
        
        # Hervorhebung bei Wahl
        line_width = 0
        # RICHTIG:
        if name == player_choice:
            line_width = 5
            size = 40
        if name == comp_choice:
            line_width = 5
            size = 40
            color = 'white' # Markierung für Computer

        fig.add_trace(go.Scatter(
            x=[x[i]], y=[y[i]],
            mode='markers+text',
            marker=dict(size=size, color=colors[name], line=dict(width=line_width, color='black')),
            text=[name], textposition="top center",
            hoverinfo='text'
        ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- SPIEL-FUNKTIONEN ---
def determine_winner(player, computer):
    rules = {
        'Rot': ['Grün', 'Gelb'], 'Gelb': ['Blau', 'Lila'], 'Blau': ['Grün', 'Rot'],
        'Grün': ['Gelb', 'Lila'], 'Lila': ['Rot', 'Blau']
    }
    if player == computer: return "draw"
    return "player" if computer in rules.get(player, []) else "computer"

# --- UI LOGIK ---
if 'p_score' not in st.session_state:
    st.session_state.update({'p_score': 0, 'c_score': 0, 'over': False})

st.title("🏹 KI Farb-Pentagon Battle")

# Scoreboard
sc1, sc2, sc3 = st.columns([2,1,2])
sc1.metric("Spieler", st.session_state.p_score)
sc2.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
sc3.metric("Computer", st.session_state.c_score)

# Pentagon Anzeige
st.plotly_chart(draw_interactive_pentagon(), use_container_width=True, config={'displayModeBar': False})

if not st.session_state.over:
    img_file = st.camera_input("Scanne deine Farbe")
    
    if img_file:
        model, class_names = load_model_and_labels()
        img = Image.open(img_file).convert("RGB")
        
        # Prediction
        size = (224, 224)
        img_res = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        img_arr = (np.asarray(img_res).astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = img_arr
        
        pred = model.predict(data, verbose=0)
        p_choice = class_names[np.argmax(pred)]
        conf = np.max(pred)
        
        c_choice = np.random.choice(['Rot', 'Gelb', 'Grün', 'Blau', 'Lila'])
        
        # Ergebnis-Visualisierung
        st.divider()
        st.plotly_chart(draw_interactive_pentagon(p_choice, c_choice), use_container_width=True)
        
        res = determine_winner(p_choice, c_choice)
        if res == "player":
            st.success(f"PUNKT! {p_choice} schlägt {c_choice}")
            st.session_state.p_score += 1
        elif res == "computer":
            st.error(f"VERLOREN! {c_choice} schlägt {p_choice}")
            st.session_state.c_score += 1
        else:
            st.info("Unentschieden!")

        if st.session_state.p_score >= 4 or st.session_state.c_score >= 4:
            st.session_state.over = True
            st.rerun()
else:
    if st.session_state.p_score >= 4:
        st.balloons()
        st.success("# 🏆 DU HAST GEWONNEN!")
    else:
        st.error("# 🤖 KI HAT GEWONNEN!")
    
    if st.button("Neustart"):
        st.session_state.p_score = 0
        st.session_state.c_score = 0
        st.session_state.over = False
        st.rerun()
