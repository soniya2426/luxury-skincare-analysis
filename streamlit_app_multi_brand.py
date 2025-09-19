
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Multi-Brand Luxury Skincare Dashboard", layout="wide")

@st.cache_data
def load_excel(path="ibr final responses for dashboard 2.xlsx"):
    # load all sheets into a dict of dataframes (strip column names)
    xls = pd.ExcelFile(path)
    sheets = {}
    for s in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=s)
        df.columns = [c.strip() for c in df.columns]
        sheets[s.strip()] = df
    return sheets

sheets = load_excel()

st.title("Multi‑Brand Luxury Skincare Dashboard")
st.markdown("Select a brand (sheet) to explore — or compare two brands side-by-side.")

mode = st.sidebar.selectbox("View mode", ["Single brand", "Compare brands"])

def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def compute_kmeans(df, n_clusters=3):
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num.shape[1] < 2 or df.shape[0] < n_clusters:
        return None, None, None
    X = num.fillna(num.mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(Xs)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    res = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)
    res["cluster"] = kmeans.labels_.astype(str)
    return res, kmeans, num.columns.tolist()

if mode == "Single brand":
    brand = st.sidebar.selectbox("Choose brand / sheet", list(sheets.keys()))
    df = sheets[brand].copy()
    st.header(f"Brand: {brand} — Overview")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    # filters
    with st.expander("Filters"):
        age_vals = df.get("age group", pd.Series()).unique().tolist() if "age group" in df.columns else []
        live_vals = df.get("live in uae", pd.Series()).unique().tolist() if "live in uae" in df.columns else []
        time_vals = df.get("time of using luxury skincare", pd.Series()).unique().tolist() if "time of using luxury skincare" in df.columns else []
        sel_age = st.multiselect("Age group", options=age_vals, default=age_vals)
        sel_live = st.multiselect("Live in UAE", options=live_vals, default=live_vals)
        sel_time = st.multiselect("Time using luxury skincare", options=time_vals, default=time_vals)
    if "age group" in df.columns and sel_age: df = df[df["age group"].isin(sel_age)]
    if "live in uae" in df.columns and sel_live: df = df[df["live in uae"].isin(sel_live)]
    if "time of using luxury skincare" in df.columns and sel_time: df = df[df["time of using luxury skincare"].isin(sel_time)]
    st.subheader("Key metrics")
    numcols = numeric_cols(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample size", df.shape[0])
    if "purchase intention" in df.columns:
        col2.metric("Purchase intention (mean)", round(df["purchase intention"].mean(), 3))
    if "would recommend" in df.columns:
        col3.metric("Would recommend (mean)", round(df["would recommend"].mean(), 3))
    if "brand loyalty" in df.columns:
        col4.metric("Brand loyalty (mean)", round(pd.to_numeric(df["brand loyalty"], errors='coerce').mean(), 3))

    st.subheader("Demographics & distributions")
    left, right = st.columns(2)
    if "age group" in df.columns:
        left.write("Age group counts")
        left.bar_chart(df["age group"].value_counts())
    if "live in uae" in df.columns:
        right.write("Live in UAE counts")
        right.bar_chart(df["live in uae"].value_counts())

    st.subheader("Attribute means (numeric columns)")
    if len(numcols)>0:
        st.dataframe(df[numcols].mean().round(3))

    st.subheader("Correlation matrix")
    if len(numcols)>1:
        corr = df[numcols].corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough numeric columns for correlation matrix.")

    st.subheader("Segmentation (KMeans + PCA)")
    res, kmeans, used_cols = compute_kmeans(df, n_clusters=3)
    if res is None:
        st.write("Not enough numeric data to run clustering for this selection.")
    else:
        plotdf = res.copy()
        plotdf = plotdf.join(df[used_cols])
        fig = px.scatter(plotdf, x="PC1", y="PC2", color="cluster", hover_data=used_cols, title="PCA projection colored by cluster")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plotdf.groupby("cluster").size().rename("count").to_frame())

    st.subheader("Raw data (first 50 rows)")
    st.dataframe(df.head(50))

    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode('utf-8'), file_name=f"{brand.replace(' ','_')}_filtered.csv")

else:
    # Compare brands
    st.header("Compare two brands")
    choices = list(sheets.keys())
    a = st.sidebar.selectbox("Brand A", choices, index=0)
    b = st.sidebar.selectbox("Brand B", choices, index=min(1, len(choices)-1))
    st.write(f"Comparing **{a}** vs **{b}**")
    df_a = sheets[a].copy()
    df_b = sheets[b].copy()
    # Basic KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{a} size", df_a.shape[0])
    col2.metric(f"{b} size", df_b.shape[0])
    if "purchase intention" in df_a.columns and "purchase intention" in df_b.columns:
        col1.metric(f"{a} purchase intention", round(df_a["purchase intention"].mean(),3))
        col2.metric(f"{b} purchase intention", round(df_b["purchase intention"].mean(),3))
        diff = round(df_a["purchase intention"].mean() - df_b["purchase intention"].mean(),3)
        col3.metric("Difference (A - B)", diff)
    st.subheader("Side-by-side attribute means (numeric columns)")
    na = df_a.select_dtypes(include=[np.number]).mean().rename(a)
    nb = df_b.select_dtypes(include=[np.number]).mean().rename(b)
    merged = pd.concat([na, nb], axis=1).round(3)
    st.dataframe(merged)

    st.subheader("Compare distributions")
    if "age group" in df_a.columns or "age group" in df_b.columns:
        fig = px.histogram(pd.DataFrame({
            "age_group": pd.concat([df_a.get("age group", pd.Series()), df_b.get("age group", pd.Series())], ignore_index=True),
            "brand": [a]*df_a.shape[0] + [b]*df_b.shape[0]
        }), x="age_group", color="brand", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation comparison (side-by-side heatmaps)")
    ca = df_a.select_dtypes(include=[np.number]).corr()
    cb = df_b.select_dtypes(include=[np.number]).corr()
    if not ca.empty:
        st.write(f"Correlation for {a}")
        st.plotly_chart(px.imshow(ca, text_auto=True), use_container_width=True)
    if not cb.empty:
        st.write(f"Correlation for {b}")
        st.plotly_chart(px.imshow(cb, text_auto=True), use_container_width=True)

    st.subheader("Clustering comparison (PCA scatter per brand)")
    res_a, _, _ = compute_kmeans(df_a, n_clusters=3)
    res_b, _, _ = compute_kmeans(df_b, n_clusters=3)
    cols = st.columns(2)
    if res_a is not None:
        cols[0].write(f"{a} PCA clusters")
        cols[0].plotly_chart(px.scatter(res_a, x="PC1", y="PC2", color="cluster"), use_container_width=True)
    else:
        cols[0].write(f"Not enough numeric data in {a} for clustering.")
    if res_b is not None:
        cols[1].write(f"{b} PCA clusters")
        cols[1].plotly_chart(px.scatter(res_b, x="PC1", y="PC2", color="cluster"), use_container_width=True)
    else:
        cols[1].write(f"Not enough numeric data in {b} for clustering.")

    st.subheader("Download raw sheets")
    st.download_button(f"Download {a} sheet (CSV)", df_a.to_csv(index=False).encode('utf-8'), file_name=f"{a.replace(' ','_')}.csv")
    st.download_button(f"Download {b} sheet (CSV)", df_b.to_csv(index=False).encode('utf-8'), file_name=f"{b.replace(' ','_')}.csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Built for: multi-brand exploration. You can change the Excel filename inside the code if you prefer to use `processed_data.csv` per sheet.")
