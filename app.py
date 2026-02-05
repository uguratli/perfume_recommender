import streamlit as st
import pandas as pd
import base64
import time
from recommender import recommend
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

# -----------------------------
# 🎥 Banner Video
# -----------------------------
def show_top_video(video_file):
    with open(video_file, "rb") as f:
        video_bytes = f.read()
        encoded = base64.b64encode(video_bytes).decode()

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
        height: 65vh;
        left: 50%;
        margin-left: -50vw;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        background: transparent;
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
# 🚀 Page Setup
# -----------------------------
st.set_page_config(page_title="Perfume Recommender", page_icon="🌸", layout="centered")

# Session state
if "searched" not in st.session_state:
    st.session_state.searched = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "search_time" not in st.session_state:
    st.session_state.search_time = None

# -----------------------------
# ⏳ Auto Reset After 30 Seconds
# -----------------------------
if st.session_state.searched and st.session_state.search_time:
    if time.time() - st.session_state.search_time > 30:
        st.session_state.clear()
        st.rerun()

# -----------------------------
# 🎬 Banner Switch
# -----------------------------
DEFAULT_BANNER = "background1.mp4"
RESULT_BANNER = "background2.mp4"
video_to_show = RESULT_BANNER if st.session_state.searched else DEFAULT_BANNER
show_top_video(video_to_show)

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
        if query.strip():
            st.session_state.searched = True
            st.session_state.last_query = query
            st.session_state.search_time = time.time()
            st.rerun()
        else:
            st.warning("Please describe a scent or mood.")

# -----------------------------
# 📊 Results
# -----------------------------
if st.session_state.searched and st.session_state.last_query:
    remaining = 30 - int(time.time() - st.session_state.search_time) # type: ignore
    if remaining > 0:
        st.caption(f"🔄 Resetting in {remaining} seconds...")

    with st.spinner("Blending fragrance notes..."):
        results_df, _ = recommend(st.session_state.last_query)

    if results_df.empty:
        st.error("No matches found. Try another description.")
    else:
        st.success(f"Found {len(results_df)} perfumes ✨")
        st.dataframe(results_df, use_container_width=True)
