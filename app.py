import streamlit as st
import pandas as pd
import base64

# -----------------------------
# 🔮 Dummy Recommender (replace with yours)
# -----------------------------
def recommend_perfumes(query: str) -> pd.DataFrame:
    data = {
        "Perfume": ["Chanel Chance", "Dior Sauvage", "YSL Libre"],
        "Brand": ["Chanel", "Dior", "Yves Saint Laurent"],
        "Main Notes": ["Floral, Citrus", "Bergamot, Pepper", "Lavender, Vanilla"],
        "Match Score": [0.91, 0.87, 0.85]
    }
    return pd.DataFrame(data)

def play_background_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
        encoded = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio id="bgAudio" loop>
        <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
    </audio>

    <script>
    const audio = document.getElementById("bgAudio");

    function startAudio() {{
        if (audio.paused) {{
            audio.volume = 0.3;
            audio.play();
        }}
        document.removeEventListener("click", startAudio);
    }}

    document.addEventListener("click", startAudio);
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# -----------------------------
# 🎥 Top Banner Video
# -----------------------------
import time

def show_top_video(video_file):
    with open(video_file, "rb") as f:
        video_bytes = f.read()
        encoded = base64.b64encode(video_bytes).decode()

    # Unique ID prevents browser caching
    unique_id = str(time.time()).replace(".", "")

    video_html = f"""
    <style>
    .block-container {{
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }}

    .video-wrapper {{
        position: relative;
        width: 100vw;
        height: 70vh;
        left: 50%;
        margin-left: -50vw;
        overflow: hidden;
        background: transparent;
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .video-wrapper video {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }}
    </style>

    <div class="video-wrapper">
        <video autoplay muted loop key="{unique_id}">
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)


# -----------------------------
# 🖼 Background Image Switch
# -----------------------------
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# ⏳ Auto Refresh (30s)
# -----------------------------
def auto_refresh():
    st.markdown(
        """
        <script>
        setTimeout(function(){
            window.location.reload();
        }, 30000);
        </script>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# 🚀 Page Setup
# -----------------------------
st.set_page_config(page_title="Perfume Recommender", page_icon="🌸", layout="centered")

# Track search state
if "searched" not in st.session_state:
    st.session_state.searched = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

    

# Backgrounds
DEFAULT_BG = "https://images.unsplash.com/photo-1528747045269-390fe33c19c9"
RESULT_BG = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
DEFAULT_BANNER = "background1.mp4"
RESULT_BANNER = "background2.mp4"
#set_bg_image(RESULT_BG if st.session_state.searched else DEFAULT_BG)

# -----------------------------
# 🎥 Show Video Banner FIRST
# -----------------------------
#show_top_video("background1.mp4")   # put video file in same folder
video_to_show = RESULT_BANNER if st.session_state.searched else DEFAULT_BANNER
show_top_video(video_to_show)# -----------------------------
play_background_audio("background_snd.mp3")
audio_file = open("background_snd.mp3", "rb")
audio_bytes = audio_file.read()

st.audio(audio_bytes, format="audio/mp3", start_time=0)

# -----------------------------
# 🌸 App Title
# -----------------------------
st.title("Perfume Recommender")
st.write("Describe a scent, mood, season, or perfume you love.")

# -----------------------------
# 🔎 Search Form
# -----------------------------
with st.form("perfume_form"):
    query = st.text_input("What kind of perfume are you looking for?")
    submitted = st.form_submit_button("Find Perfumes")

    if submitted:
        if query.strip() == "":
            st.warning("Please describe a scent or mood.")
        else:
            st.session_state.searched = True
            st.session_state.last_query = query
            st.rerun()

# -----------------------------
# 📊 Show Results After Search
# -----------------------------
if st.session_state.searched and st.session_state.last_query:
    with st.spinner("Blending fragrance notes..."):
        results_df = recommend_perfumes(st.session_state.last_query)

    if results_df.empty:
        st.error("No matches found. Try another description.")
    else:
        st.success(f"Found {len(results_df)} perfumes ✨")
        st.dataframe(results_df, use_container_width=True)

# -----------------------------
# ⏳ Auto Reset After 30 Seconds
# -----------------------------
st.markdown(
    """
    <script>
    setTimeout(function(){
        window.location.reload();
    }, 30000);
    </script>
    """,
    unsafe_allow_html=True
)