import streamlit as st
import pandas as pd

# Import your recommender function
from recommender import recommend   # change to your file name

st.set_page_config(page_title="Perfume Recommender", page_icon="🌸")

st.title("🌸 Perfume Recommender")
st.write("Describe a scent you like, a vibe, notes, or a favorite perfume.")

# --- User Input ---
query = st.text_input("What kind of perfume are you looking for?")

# --- Button to trigger recommendation ---
if st.button("Find Perfumes"):
    if query.strip() == "":
        st.warning("Please enter something to describe your scent preferences.")
    else:
        with st.spinner("Sniffing out recommendations..."):
            results_df, _ = recommend(query)

        if results_df is None or results_df.empty:
            st.error("No recommendations found. Try a different description.")
        else:
            st.success(f"Found {len(results_df)} perfumes for you ✨")
            st.dataframe(results_df, use_container_width=True)