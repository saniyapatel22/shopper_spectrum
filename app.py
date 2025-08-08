import os
import gdown
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------- Session State Initialization -----------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_home():
    st.session_state.page = 'home'

def go_to_recommendation():
    st.session_state.page = 'recommendation'

def go_to_segmentation():
    st.session_state.page = 'segmentation'

# ----------------- Load Dataset -----------------
@st.cache_data
def load_data():
    file_id = '18ndn0jnsELquDYxLLplD4stUcc8icjKr'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'online_retail.csv'

    # Download only if file does not exist
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)


    data = pd.read_csv(output,encoding='ISO-8859-1',on_bad_lines="skip")
    
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])  
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    return data

data = load_data()

# ----------------- Sidebar Navigation -----------------
st.sidebar.title("📊 Navigation")
st.sidebar.button("🏠 Home", on_click=go_home)
st.sidebar.button("🛍️ Product Recommendation", on_click=go_to_recommendation)
st.sidebar.button("👥 Customer Segmentation", on_click=go_to_segmentation)

# ----------------- Main View -----------------
st.title(" Shopper Spectrum Dashboard")

if st.session_state.page == 'home':
    st.subheader("Welcome!")
    st.markdown("Explore two main features:")
    st.markdown("**Product Recommendation** - Get top 5 similar products based on purchase patterns.")
    st.markdown("**Customer Segmentation** - Discover which segment a customer belongs to using RFM values.")

elif st.session_state.page == 'recommendation':
    st.header(" Product Recommendation")

    # Pivot table (Customer x Product)
    pivot = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    cosine_sim = cosine_similarity(pivot.T)
    similarity_df = pd.DataFrame(cosine_sim, index=pivot.columns, columns=pivot.columns)

    stock_to_desc = data[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
    desc_to_stock = {v: k for k, v in stock_to_desc.items()}

    product_name = st.text_input("Enter Product Name:")

    if st.button("Get Recommendations"):
        if product_name in desc_to_stock:
            prod_code = desc_to_stock[product_name]
            sim_scores = similarity_df[prod_code].sort_values(ascending=False)[1:6]
            st.success("Top 5 similar products:")
            for i, (code, score) in enumerate(sim_scores.items(), 1):
                name = stock_to_desc.get(code, "Unknown")
                st.markdown(f"**{i}.** {name} (`{code}`)")
        else:
            st.error("Product not found! Try a valid product name.")

elif st.session_state.page == 'segmentation':
    st.header(" Customer Segmentation (RFM-Based)")
    st.subheader("Enter RFM Values")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spent)", min_value=0.0, value=500.0)

    if st.button("Predict Cluster"):
        rfm_input = np.array([[recency, frequency, monetary]])

        # Calculate RFM from dataset
        rfm_data = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (pd.Timestamp('2011-12-10') - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()

        rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Scale and train KMeans
        scaler = StandardScaler()
        scaler.fit(rfm_data[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(scaler.transform(rfm_data[['Recency', 'Frequency', 'Monetary']]))

        # Predict segment
        cluster = kmeans.predict(scaler.transform(rfm_input))[0]

        cluster_names = {
            0: "High-Value",
            1: "Occasional",
            2: "Regular",
            3: "At-Risk"
        }

        st.success(f"Predicted Segment: **{cluster_names.get(cluster, 'Unknown')}** (Cluster {cluster})")
# Navigate to your project folder (if not already there)













