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


# -----------------------------
# 🎥 Top Banner Video
# -----------------------------
def show_top_video(video_file):
    with open(video_file, "rb") as f:
        video_bytes = f.read()
        encoded = base64.b64encode(video_bytes).decode()

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
        height: 28vh;   /* banner height */
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        overflow: hidden;
        background: black;  /* fills empty space if aspect ratio differs */
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .video-wrapper video {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;  /* 👈 shows full video, no cropping */
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

# Backgrounds
DEFAULT_BG = "https://images.unsplash.com/photo-1528747045269-390fe33c19c9"
RESULT_BG = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"

set_bg_image(RESULT_BG if st.session_state.searched else DEFAULT_BG)

# -----------------------------
# 🎥 Show Video Banner FIRST
# -----------------------------
show_top_video("background1.mp4")   # put video file in same folder

# -----------------------------
# 🌸 App Content
# -----------------------------
st.title("Perfume Recommender")
st.write("Describe a scent, vibe, season, or perfume you love.")

query = st.text_input("What kind of perfume are you looking for?")

if st.button("Find Perfumes"):
    if query.strip() == "":
        st.warning("Please describe a scent or mood.")
    else:
        st.session_state.searched = True
        with st.spinner("Blending fragrance notes..."):
            results_df = recommend_perfumes(query)

        if results_df.empty:
            st.error("No matches found. Try another description.")
        else:
            st.success(f"Found {len(results_df)} perfumes for you ✨")
            st.dataframe(results_df, use_container_width=True)

# Start reset timer
auto_refresh()
