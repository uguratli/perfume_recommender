from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import normalize


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
raw_data = pd.read_csv("fra_perfumes.csv")
with open('data.pkl', 'rb') as f:
   data = pickle.load(f)

desc_embs = np.load("desc_embs.npy")
accord_embs = np.load("accord_embs.npy")
gender_embs = np.load("gender_embss.npy")
brand_embs = np.load("brand_embs.npy")

W_DESC   = 0.45
W_ACCORD = 0.25
W_GENDER = 0.20
W_BRAND = 0.10

final_embs = (
    W_DESC   * desc_embs +
    W_ACCORD * accord_embs +
    W_GENDER * gender_embs + 
    W_BRAND  * brand_embs
)

final_embs = final_embs / np.linalg.norm(final_embs, axis=1, keepdims=True)

dim = final_embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(final_embs.astype(np.float32)) # type: ignore

brands = sorted(data["Brand"].unique())
brand_emb_dict = {}

for b in brands:
    # find any row belonging to this brand
    idx = data.index[data["Brand"] == b][0]
    brand_emb_dict[b] = brand_embs[idx] / np.linalg.norm(brand_embs[idx])

mean_brand_emb = brand_embs.mean(axis=0)
mean_brand_emb = mean_brand_emb / np.linalg.norm(mean_brand_emb)

def expand_gender_vec(gv, dim=384):
    # Convert 3-dim gender vector → same dimensionality as BERT embedding
    return np.pad(gv, (0, dim - len(gv)), 'constant')

def detect_brand_in_query(text, brands):
    """ Detect if any brand from the list is mentioned in the text.
        Return the brand if found, else None."""
    t = text.lower()
    for b in brands:
        if b.lower() in t:
            return b
    return None

def detect_gender_in_query(text):
    """Detech gender from the text.
       Return a 2D numpy array representing the gender embedding."""
    t = text.lower()
    if "unisex" in t:
        return np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    if "men" in t or "male" in t:
        return np.array([1.0, 0.0])
    if "women" in t or "female" in t:
        return np.array([0.0, 1.0])
    # default neutral
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)])

def clean_query_text(text, brands):
    """Remove known brand names & gender words so BERT focuses on scent description."""
    t = text.lower()
    for b in brands:
        t = t.replace(b.lower(), "")
    t = t.replace("unisex", "")
    t = t.replace("men", "")
    t = t.replace("man", "")
    t = t.replace("male", "")
    t = t.replace("women", "")
    t = t.replace("woman", "")
    t = t.replace("female", "")
    return t.strip()

def embed_query(text, model, brands, mean_brand_emb):
    # ----- BRAND -----
    brand = detect_brand_in_query(text, brands)
    if brand:
        brand_emb = brand_emb_dict[brand]     # PCA-reduced or SBERT brand embedding
    else:
        brand_emb = mean_brand_emb            # neutral brand

    # ----- GENDER -----
    gender_vec = detect_gender_in_query(text)
    gender_emb = expand_gender_vec(gender_vec)   # your 2-dim → 384-dim function

    # ----- CLEAN TEXT FOR DESCRIPTION + ACCORD -----
    cleaned = clean_query_text(text, brands)

    # ----- SBERT DESCRIPTION + ACCORD ENCODING -----
    desc_emb   = model.encode([cleaned])[0]
    accord_emb = model.encode([cleaned])[0]
    
    desc_emb   = normalize(desc_emb.reshape(1, -1))[0]
    accord_emb = normalize(accord_emb.reshape(1, -1))[0]
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

def recommend(query, model=model, brands=brands, mean_brand_emb=mean_brand_emb, faiss_index=index, data=data, top_k=10):
    q_emb = embed_query(query, model, brands, mean_brand_emb)
    
    D, I = faiss_index.search(q_emb.reshape(1, -1), top_k)  # type: ignore
    
    results = []
    for idx, dist in zip(I[0], D[0]):
        row = data.iloc[idx]
        results.append({
            "Name": row["Name"],
            "Brand": row["Brand"],
            "Gender": row["Gender"],
            "Main Accords": raw_data["Main Accords"][idx],
            "Description": raw_data["Description"][idx],
            "Score": row['Weighted Rating'],
            'Distance': float(dist),
            'URL': raw_data["url"][idx]
        })
    results_df = pd.DataFrame(results)
    results_df["Main Accords"] = results_df["Main Accords"].apply(lambda x: x.replace("[]", "Not Specified"))
    results_df["Main Accords"] = results_df["Main Accords"].apply(lambda x: x.replace("]", "").replace("[", "").replace("'", "").replace(",", ", "))
    bottom_limit = results_df['Distance'].quantile(.5)  # Filter out results that are below the 50th percentile in similarity
    results_df = results_df[results_df['Distance'] >= bottom_limit].sort_values('Score', ascending=False)
    return results_df, results