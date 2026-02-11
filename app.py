import streamlit as st
import pandas as pd
import base64
import time
from streamlit_autorefresh import st_autorefresh
from recommender import recommend


# -----------------------------
# 🎥 Banner Video
# -----------------------------
def show_top_video(video_file):
    with open(video_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    video_html = f"""
    <style>
    .block-container {{
        padding: 0;
    }}
    .video-wrapper {{
        width: 100vw;
        height: 65vh;
        margin-left: -50vw;
        left: 50%;
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        background: black;
    }}
    video {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }}
    </style>

    <div class="video-wrapper">
        <video autoplay muted loop>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# -----------------------------
# 🧼 Reset Helper
# -----------------------------
def reset_state():
    st.session_state.searched = False
    st.session_state.last_query = ""
    st.session_state.search_time = None

# -----------------------------
# 🚀 Page Setup
# -----------------------------
st.set_page_config(page_title="Perfume Recommender", page_icon="🌸")

# -----------------------------
# 🧠 Session State Init
# -----------------------------
if "searched" not in st.session_state:
    reset_state()

# -----------------------------
# ⏳ Auto-refresh every second (ONLY after search)
# -----------------------------
if st.session_state.searched:
    st_autorefresh(interval=1000, key="timer")

# -----------------------------
# ⏳ Auto Reset after 30s
# -----------------------------
if st.session_state.searched and st.session_state.search_time:
    if time.time() - st.session_state.search_time > 30:
        reset_state()
        st.rerun()

# -----------------------------
# 🎬 Banner Switch
# -----------------------------
DEFAULT_BANNER = "background1.mp4"
RESULT_BANNER = "background2.mp4"
show_top_video(RESULT_BANNER if st.session_state.searched else DEFAULT_BANNER)

# -----------------------------
# 🌸 App Content
# -----------------------------
st.title("Perfume Recommender")
st.write("Describe a scent, mood, season, or perfume you love.")

# -----------------------------
# 🔎 Search Form
# -----------------------------
with st.form("perfume_form"):
    query = st.text_input("What kind of perfume are you looking for?")
    submitted = st.form_submit_button("Find Perfumes")

    if submitted and query.strip():
        st.session_state.searched = True
        st.session_state.last_query = query
        st.session_state.search_time = time.time()
        st.rerun()

    elif submitted:
        st.warning("Please describe a scent or mood.")

# -----------------------------
# 📊 Results
# -----------------------------
if st.session_state.searched and st.session_state.last_query:
    remaining = 30 - int(time.time() - st.session_state.search_time)
    st.caption(f"🔄 Resetting in {remaining}s")

    with st.spinner("Blending fragrance notes..."):
        results_df, _ = recommend(st.session_state.last_query)

    st.dataframe(results_df[['Name', 'Brand', 'Gender', 'Main Accords', 'Description', 'Score', 'URL']].set_index('Name'), use_container_width=True)

