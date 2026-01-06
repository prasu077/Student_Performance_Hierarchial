import streamlit as st
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Student Performance Segmentation",
    page_icon="🎓",
    layout="wide"
)

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Performance Segmentation</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Hierarchical Clustering using Academic Performance Data</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------ FILE UPLOAD (FIXED) ------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Performance CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------ DATA PREPROCESSING ------------------
numeric_df = df.select_dtypes(include="number")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Clustering Settings")

n_clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=6,
    value=4
)

# ------------------ DENDROGRAM ------------------
st.subheader("🌳 Dendrogram")

fig, ax = plt.subplots(figsize=(10, 4))
sch.dendrogram(sch.linkage(X_scaled, method="ward"), ax=ax)
ax.set_xlabel("Students")
ax.set_ylabel("Distance")
st.pyplot(fig)

# ------------------ CLUSTERING ------------------
hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward"
)

clusters = hc.fit_predict(X_scaled)
df["Cluster"] = clusters

# ------------------ SILHOUETTE SCORE ------------------
score = silhouette_score(X_scaled, clusters)
st.success(f"🔹 Silhouette Score: **{score:.3f}**")

# ------------------ CLUSTER SUMMARY ------------------
st.subheader("📊 Cluster Summary (Mean Values)")
st.dataframe(
    df.groupby("Cluster").mean(numeric_only=True),
    use_container_width=True
)

# ------------------ VISUALIZATION ------------------
st.subheader("📈 Cluster Visualization")

col1, col2 = numeric_df.columns[:2]

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    x=df[col1],
    y=df[col2],
    hue=df["Cluster"],
    palette="Set2",
    ax=ax2
)
ax2.set_xlabel(col1)
ax2.set_ylabel(col2)
ax2.set_title("Student Clusters")
st.pyplot(fig2)

# ------------------ INTERPRETATION ------------------
st.subheader("🧠 Cluster Interpretation")

st.markdown("""
- **Cluster 0** → High performance students  
- **Cluster 1** → Average performers  
- **Cluster 2** → Students needing academic support  
- **Cluster 3** → At-risk students  

*(Interpretation may vary based on dataset features)*
""")
