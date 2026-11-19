import re
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Drop rows with missing Description and reset index on both DataFrames so positional index 0..N-1 aligns
raw_data = pd.read_csv("fra_perfumes.csv").dropna(subset=["Description"]).reset_index(drop=True)
with open("data.pkl", "rb") as f:
    data = pickle.load(f).reset_index(drop=True)

UNISEX_VEC = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float32)
GENDER_VECTORS = {
    "men":    np.array([1.0, 0.0], dtype=np.float32),
    "male":   np.array([1.0, 0.0], dtype=np.float32),
    "women":  np.array([0.0, 1.0], dtype=np.float32),
    "female": np.array([0.0, 1.0], dtype=np.float32),
    "unisex": UNISEX_VEC,
}


def expand_gender_vec(gv, dim=384):
    """Convert 2D gender vector to same dimensionality as MiniLM embedding (384)."""
    return np.pad(gv, (0, dim - len(gv)), mode="constant").astype(np.float32)


desc_embs = np.load("desc_embs.npy").astype(np.float32)
accord_embs = np.load("accord_embs.npy").astype(np.float32)
gender_embs = np.load("gender_embss.npy").astype(np.float32)
brand_embs = np.load("brand_embs.npy").astype(np.float32)

# Auto-repair gender_embs in memory if the file on disk still contains the legacy [0, 0, 1] fallback
if np.any(gender_embs[:, 2] != 0):
    gender_embs = np.vstack([
        expand_gender_vec(GENDER_VECTORS.get(str(g).strip().lower(), UNISEX_VEC))
        for g in data["Gender"]
    ]).astype(np.float32)
    gender_embs = normalize(gender_embs).astype(np.float32)

W_DESC   = 0.45
W_ACCORD = 0.25
W_GENDER = 0.20
W_BRAND  = 0.10

final_embs = (
    W_DESC   * desc_embs +
    W_ACCORD * accord_embs +
    W_GENDER * gender_embs +
    W_BRAND  * brand_embs
)

final_embs = normalize(final_embs).astype(np.float32)

dim = final_embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(final_embs)  # type: ignore

brands = sorted(data["Brand"].dropna().unique())
brand_emb_dict = {}

for b in brands:
    idx = int(data.index[data["Brand"] == b][0])
    vec = brand_embs[idx]
    norm = np.linalg.norm(vec)
    brand_emb_dict[b] = (vec / norm) if norm > 0 else vec

mean_brand_emb = brand_embs.mean(axis=0)
mean_brand_emb = (mean_brand_emb / np.linalg.norm(mean_brand_emb)).astype(np.float32)

GENDER_CLEAN_PATTERN = re.compile(
    r"\b(unisex|women|woman|female|men|man|male)\b", flags=re.IGNORECASE
)


def detect_brand_in_query(text, brands_list):
    """Detect if any brand from the list is mentioned as a whole word in the text."""
    t = text.lower()
    for b in sorted(brands_list, key=len, reverse=True):
        if not b:
            continue
        if re.search(rf"\b{re.escape(b.lower())}\b", t):
            return b
    return None


def detect_gender_in_query(text):
    """Detect gender from whole words in the text and return a 2D numpy array."""
    t = text.lower()
    if re.search(r"\bunisex\b", t):
        return UNISEX_VEC
    has_women = bool(re.search(r"\b(women|woman|female)\b", t))
    has_men = bool(re.search(r"\b(men|man|male)\b", t))
    if has_women and not has_men:
        return GENDER_VECTORS["women"]
    if has_men and not has_women:
        return GENDER_VECTORS["men"]
    return UNISEX_VEC


def clean_query_text(text, detected_brand=None):
    """Remove detected brand and gender words using word boundaries so SBERT focuses on scent description."""
    t = text.lower()
    if detected_brand:
        t = re.sub(rf"\b{re.escape(detected_brand.lower())}\b", " ", t)
    t = GENDER_CLEAN_PATTERN.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else text.strip()


def embed_query(text, model, brands, mean_brand_emb):
    # ----- BRAND -----
    brand = detect_brand_in_query(text, brands)
    if brand:
        brand_emb = brand_emb_dict[brand]
    else:
        brand_emb = mean_brand_emb

    # ----- GENDER -----
    gender_vec = detect_gender_in_query(text)
    gender_emb = expand_gender_vec(gender_vec)

    # ----- CLEAN TEXT FOR DESCRIPTION + ACCORD -----
    cleaned = clean_query_text(text, detected_brand=brand)

    # ----- SBERT DESCRIPTION + ACCORD ENCODING (single pass) -----
    text_emb = model.encode([cleaned])[0]
    text_emb = normalize(text_emb.reshape(1, -1))[0]

    desc_emb   = text_emb
    accord_emb = text_emb
    brand_emb  = normalize(brand_emb.reshape(1, -1))[0]
    gender_emb = normalize(gender_emb.reshape(1, -1))[0]

    # ----- FUSION -----
    q = (
        W_DESC   * desc_emb +
        W_ACCORD * accord_emb +
        W_GENDER * gender_emb +
        W_BRAND  * brand_emb
    )

    q = normalize(q.reshape(1, -1))[0]
    return q.astype(np.float32)


def recommend(
    query,
    model=model,
    brands=brands,
    mean_brand_emb=mean_brand_emb,
    faiss_index=index,
    data=data,
    top_k=10,
):
    q_emb = embed_query(query, model, brands, mean_brand_emb)

    D, I = faiss_index.search(q_emb.reshape(1, -1), top_k)  # type: ignore

    results = []
    for idx, dist in zip(I[0], D[0]):
        idx = int(idx)
        row = data.iloc[idx]
        raw_row = raw_data.iloc[idx]
        results.append({
            "Name": row["Name"],
            "Brand": row["Brand"],
            "Gender": row["Gender"],
            "Main Accords": str(raw_row["Main Accords"]),
            "Description": str(raw_row["Description"]),
            "Score": row["Weighted Rating"],
            "Distance": float(dist),
            "URL": raw_row["url"],
        })

    results_df = pd.DataFrame(results)
    results_df["Main Accords"] = results_df["Main Accords"].apply(
        lambda x: x.replace("[]", "Not Specified")
    )
    results_df["Main Accords"] = results_df["Main Accords"].apply(
        lambda x: x.replace("]", "").replace("[", "").replace("'", "").replace(",", ", ")
    )
    bottom_limit = results_df["Distance"].quantile(0.5)
    results_df = results_df[results_df["Distance"] >= bottom_limit].sort_values(
        "Score", ascending=False
    )
    return results_df, results
