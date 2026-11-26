import os
import time
import base64
import streamlit as st
import pandas as pd

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

from recommender import (
    recommend,
    brands,
    detect_brand_in_query,
    clean_query_text,
)

# -----------------------------
# 🚀 Page Setup (Sidebar collapsed by default)
# -----------------------------
st.set_page_config(
    page_title="Perfume Recommender",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# 🎨 Custom Styling & Cached Media
# -----------------------------
DEFAULT_BANNER = "background1.mp4"
RESULT_BANNER = "background2.mp4"
ALT_BANNER = "background3.mp4"
AUDIO_FILE = "background_snd.mp3"


@st.cache_data(show_spinner=False)
def get_base64_media(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def show_top_video(video_file: str):
    encoded = get_base64_media(video_file)
    if not encoded:
        return

    video_html = f"""
    <style>
    .block-container {{
        padding-top: 0rem !important;
        max-width: 1150px;
    }}
    .video-wrapper {{
        width: 100vw;
        height: 48vh;
        margin-left: calc(-50vw + 50%);
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        background: #0a0a0c;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.5rem;
    }}
    .video-wrapper video {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}
    .perfume-card {{
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.9rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .perfume-card:hover {{
        border-color: rgba(212, 175, 55, 0.55);
    }}
    .accord-pill {{
        display: inline-block;
        background: rgba(212, 175, 55, 0.14);
        border: 1px solid rgba(212, 175, 55, 0.35);
        color: #f3e5ab;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }}
    .gender-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    </style>

    <div class="video-wrapper">
        <video autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_recommend(query_text: str, top_k: int = 10):
    results_df, _ = recommend(query_text, top_k=top_k)
    return results_df


def infer_gender_label(text: str) -> str:
    t = text.lower()
    if "unisex" in t:
        return "Unisex"
    has_women = any(w in t.split() for w in ("women", "woman", "female"))
    has_men = any(w in t.split() for w in ("men", "man", "male"))
    if has_women and not has_men:
        return "Women"
    if has_men and not has_women:
        return "Men"
    return "Unisex / Neutral"


# -----------------------------
# 🧼 Session State & Reset Helper
# -----------------------------
def reset_state():
    st.session_state.searched = False
    st.session_state.last_query = ""
    st.session_state.search_time = None
    st.session_state.results_df = None


if "searched" not in st.session_state:
    reset_state()

# -----------------------------
# ⚙️ Sidebar Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Candidate Pool Size (Top-K)",
        min_value=6,
        max_value=30,
        value=10,
        step=2,
        help="Number of top candidates retrieved from FAISS before applying the 50th-percentile similarity filter and ranking by Bayesian score.",
    )

    gender_filter = st.selectbox(
        "Gender Preference",
        options=["Auto (Detect from query)", "Women", "Men", "Unisex"],
        index=0,
    )

    view_mode = st.radio(
        "Display Mode",
        options=["✨ Card View", "📊 Table View"],
        index=0,
    )

    st.divider()
    st.subheader("⏱️ Kiosk Timer")
    auto_reset_enabled = st.toggle(
        "Auto-reset after 30s",
        value=True,
        help="Automatically resets the screen back to the initial state 30 seconds after a search.",
    )
    reset_timeout = 30

# -----------------------------
# ⏳ Auto-refresh & Auto-Reset
# -----------------------------
if st.session_state.searched and auto_reset_enabled and st.session_state.search_time:
    elapsed = time.time() - st.session_state.search_time
    if elapsed > reset_timeout:
        reset_state()
        st.rerun()
    elif HAS_AUTOREFRESH:
        st_autorefresh(interval=1000, key="kiosk_timer")

# -----------------------------
# 🎬 Banner Switch
# -----------------------------
show_top_video(RESULT_BANNER if st.session_state.searched else DEFAULT_BANNER)

# -----------------------------
# 🌸 Header & Quick Examples
# -----------------------------
st.title("🌸 Perfume Recommender")
st.write("Describe a scent, fragrance notes, mood, season, or a brand you love.")

example_queries = [
    "Fresh citrus, lime, basil and marine summer scent for men",
    "Warm vanilla, cinnamon, amber and sweet spicy winter perfume for women",
    "Woody oud, rose, leather and smoky incense unisex fragrance",
    "Clean powdery iris, white floral jasmine and soft musk for women",
]

st.caption("💡 **Quick Examples** (click to try):")
ex_cols = st.columns(len(example_queries))
selected_example = None
for col, ex in zip(ex_cols, example_queries):
    if col.button(ex, use_container_width=True):
        selected_example = ex

# -----------------------------
# 🔎 Search Form
# -----------------------------
with st.form("perfume_form", clear_on_submit=False):
    default_val = selected_example if selected_example else st.session_state.last_query
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "What kind of perfume are you looking for?",
            value=default_val,
            placeholder="e.g., Warm vanilla, amber and cinnamon scent for women...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("🔍 Find Perfumes", use_container_width=True)

# Trigger search either via form submit or clicking an example button
active_trigger_query = selected_example if selected_example else (query if submitted else None)

if active_trigger_query is not None:
    cleaned_input = active_trigger_query.strip()
    if cleaned_input:
        # Append explicit gender hint if user selected one in sidebar
        effective_query = cleaned_input
        if gender_filter == "Women" and "women" not in cleaned_input.lower():
            effective_query = f"{cleaned_input} women"
        elif gender_filter == "Men" and "men" not in cleaned_input.lower():
            effective_query = f"{cleaned_input} men"
        elif gender_filter == "Unisex" and "unisex" not in cleaned_input.lower():
            effective_query = f"{cleaned_input} unisex"

        with st.spinner("Blending fragrance notes..."):
            df_res = cached_recommend(effective_query, top_k=top_k)

        st.session_state.searched = True
        st.session_state.last_query = cleaned_input
        st.session_state.effective_query = effective_query
        st.session_state.search_time = time.time()
        st.session_state.results_df = df_res
        st.rerun()
    elif submitted:
        st.warning("Please describe a scent, note, or mood.")

# -----------------------------
# 📊 Results Section
# -----------------------------
if st.session_state.searched and st.session_state.last_query:
    effective_q = st.session_state.get("effective_query", st.session_state.last_query)
    results_df = cached_recommend(effective_q, top_k=top_k)

    # Top status bar: query analysis + timer + reset button
    top_left, top_right = st.columns([4, 1])

    detected_brand = detect_brand_in_query(effective_q, brands)
    detected_gender = infer_gender_label(effective_q)
    cleaned_scent = clean_query_text(effective_q, detected_brand=detected_brand)

    with top_left:
        st.markdown(
            f"**Query Analysis:** &nbsp; "
            f"🎯 **Gender:** `{detected_gender}` &nbsp;|&nbsp; "
            f"🏷️ **Brand:** `{detected_brand if detected_brand else 'Neutral (All Brands)'}` &nbsp;|&nbsp; "
            f"🌿 **Scent Vector Text:** `{cleaned_scent}`"
        )
        if auto_reset_enabled and st.session_state.search_time:
            remaining = max(0, reset_timeout - int(time.time() - st.session_state.search_time))
            st.caption(f"🔄 Resetting in **{remaining}s**")

    with top_right:
        if st.button("🔄 Reset Search", use_container_width=True):
            reset_state()
            st.rerun()

    st.divider()

    if results_df is None or results_df.empty:
        st.info("No matching perfumes found. Please try different notes.")
    elif view_mode == "✨ Card View":
        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            accords_list = [a.strip() for a in str(row["Main Accords"]).split(",") if a.strip()]
            accords_html = "".join(
                f'<span class="accord-pill">{a}</span>' for a in accords_list
            )
            similarity_pct = max(0.0, min(100.0, float(row["Distance"]) * 100.0))
            score_val = float(row["Score"])
            gender_val = str(row["Gender"]).capitalize()

            st.markdown(
                f"""
                <div class="perfume-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                        <div>
                            <span style="font-size:1.2rem; font-weight:700;">#{rank} &nbsp; {row['Name']}</span>
                            <span style="opacity:0.75; font-size:0.95rem; margin-left:8px;">by <b>{row['Brand']}</b></span>
                            <span class="gender-badge" style="margin-left:10px; background:rgba(255,255,255,0.1);">{gender_val}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-weight:700; color:#f5c518;">⭐ {score_val:.2f} / 5.0</span>
                            &nbsp;&bull;&nbsp;
                            <span style="font-size:0.9rem; opacity:0.85;">🎯 Match: <b>{similarity_pct:.1f}%</b></span>
                            &nbsp;&bull;&nbsp;
                            <a href="{row['URL']}" target="_blank" style="text-decoration:none; font-weight:600;">Fragrantica ↗</a>
                        </div>
                    </div>
                    <div style="margin-top:10px;">
                        {accords_html}
                    </div>
                    <div style="margin-top:8px; font-size:0.92rem; opacity:0.88; line-height:1.45;">
                        {row['Description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        display_df = results_df[
            ["Name", "Brand", "Gender", "Main Accords", "Score", "Distance", "Description", "URL"]
        ].copy()
        display_df["Distance"] = (display_df["Distance"] * 100).round(1)

        st.dataframe(
            display_df.set_index("Name"),
            use_container_width=True,
            column_config={
                "Brand": st.column_config.TextColumn("Brand"),
                "Gender": st.column_config.TextColumn("Gender"),
                "Main Accords": st.column_config.TextColumn("Main Accords"),
                "Score": st.column_config.NumberColumn(
                    "Weighted Score",
                    format="⭐ %.2f",
                ),
                "Distance": st.column_config.ProgressColumn(
                    "Match (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "URL": st.column_config.LinkColumn("Fragrantica URL", display_text="View ↗"),
            },
        )
