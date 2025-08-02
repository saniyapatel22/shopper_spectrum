import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from mpl_toolkits.mplot3d import Axes3D

# ------------------ Setup ------------------
# Create folder for visuals
output_dir = "visuals"
os.makedirs(output_dir, exist_ok=True)

# Helper function to save and show plots
def quick_show(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# ------------------ Load and Preprocess Data ------------------
df = pd.read_csv(r"C:\Users\patel\Desktop\INTERNSHIP\shopper_spectrum\online_retail.csv", encoding='ISO-8859-1')

# Drop missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Remove canceled orders
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Filter for positive quantity and price
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ------------------ 1. Transactions by Country ------------------
country_sales = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
plt.figure(figsize=(12,6))
country_sales.plot(kind='bar', color='skyblue')
plt.title('Number of Transactions by Country')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=90)
quick_show("1_transactions_by_country.png")

# ------------------ 2. Top-Selling Products ------------------
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Best-Selling Products')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product')
quick_show("2_top_selling_products.png")

# ------------------ 3. Monthly Purchase Trends ------------------
df.set_index('InvoiceDate', inplace=True)
monthly_sales = df.resample('M')['Quantity'].sum()
plt.figure(figsize=(12,6))
monthly_sales.plot()
plt.title('Monthly Purchase Trends')
plt.xlabel('Month')
plt.ylabel('Quantity Purchased')
plt.grid(True)
quick_show("3_monthly_purchase_trends.png")
df.reset_index(inplace=True)

# ------------------ 4. Transaction & Customer Spend Distributions ------------------
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Per transaction
transaction_amount = df.groupby('InvoiceNo')['TotalAmount'].sum()
plt.figure(figsize=(10,5))
sns.histplot(transaction_amount, bins=50, kde=True, color='orange')
plt.title('Monetary Value per Transaction')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
quick_show("4a_transaction_amount_distribution.png")

# Per customer
customer_amount = df.groupby('CustomerID')['TotalAmount'].sum()
plt.figure(figsize=(10,5))
sns.histplot(customer_amount, bins=50, kde=True, color='green')
plt.title('Monetary Value per Customer')
plt.xlabel('Customer Spend')
plt.ylabel('Number of Customers')
quick_show("4b_customer_amount_distribution.png")

# ------------------ 5. RFM Analysis ------------------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# Distribution plots
fig, axs = plt.subplots(1, 3, figsize=(18,5))
sns.histplot(rfm['Recency'], bins=30, kde=True, ax=axs[0], color='royalblue')
axs[0].set_title('Recency Distribution')
sns.histplot(rfm['Frequency'], bins=30, kde=True, ax=axs[1], color='seagreen')
axs[1].set_title('Frequency Distribution')
sns.histplot(rfm['Monetary'], bins=30, kde=True, ax=axs[2], color='darkorange')
axs[2].set_title('Monetary Distribution')
quick_show("5_rfm_distributions.png")

# ------------------ 6. Elbow Method ------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.grid(True)
quick_show("6_elbow_method.png")

# ------------------ 7. Clustering and Heatmap ------------------
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_profile = rfm.groupby('Cluster').mean()
plt.figure(figsize=(10,6))
sns.heatmap(cluster_profile, annot=True, cmap='Blues', fmt='.1f')
plt.title('Customer Segment Profiles (Average RFM per Cluster)')
quick_show("7_rfm_cluster_profiles_heatmap.png")

# ------------------ 8. Product Similarity Heatmap ------------------
pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
similarity_matrix = cosine_similarity(pivot.T)
similarity_df = pd.DataFrame(similarity_matrix, index=pivot.columns, columns=pivot.columns)

# Smaller sample for visualization
sample_codes = similarity_df.sample(50, axis=0).sample(50, axis=1)
plt.figure(figsize=(12,10))
sns.heatmap(sample_codes, cmap='coolwarm')
plt.title("Product Similarity Matrix (Cosine Similarity)")
quick_show("8_product_similarity_heatmap.png")

# ------------------ 9. Refined RFM Data ------------------
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]
rfm_values = rfm[['Recency', 'Frequency', 'Monetary']]
rfm_scaled = scaler.fit_transform(rfm_values)

# ------------------ 10. Elbow + Silhouette Scores ------------------
inertias, silhouettes = [], []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, kmeans.labels_))

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K, silhouettes, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
quick_show("10_elbow_and_silhouette.png")

# ------------------ 11. Final Clustering & Segment Labeling ------------------
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
segment_map = {
    0: 'Regular',
    1: 'At-Risk',
    2: 'High-Value',
    3: 'Occasional'
}
rfm['Segment'] = rfm['Cluster'].map(segment_map)

# ------------------ 12. 2D Scatter Plot ------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set2', s=100)
plt.title('Customer Segments: Recency vs Monetary')
plt.xlabel('Recency (days)')
plt.ylabel('Monetary (Total Spend)')
plt.grid(True)
quick_show("12_scatter_recency_vs_monetary.png")

# ------------------ 13. 3D RFM Scatter Plot ------------------
colors = {'High-Value': 'green', 'Regular': 'blue', 'Occasional': 'orange', 'At-Risk': 'red'}
rfm['Color'] = rfm['Segment'].map(colors)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
           c=rfm['Color'], s=50, alpha=0.6)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('3D RFM Cluster Visualization')
plt.savefig(os.path.join(output_dir, "13_3d_rfm_clusters.png"))
plt.show()

# ------------------ 14. Save Models and Segments ------------------
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
rfm.to_csv("rfm_segmented_customers.csv", index=False)

# ------------------ 15. Item-Based Collaborative Filtering ------------------
customer_product_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum'
).fillna(0)

product_similarity = cosine_similarity(customer_product_matrix.T)
similarity_df = pd.DataFrame(product_similarity,
                             index=customer_product_matrix.columns,
                             columns=customer_product_matrix.columns)

def recommend_products(product_code, top_n=5):
    if product_code not in similarity_df.columns:
        return ["Product code not found."]
    similar_scores = similarity_df[product_code].sort_values(ascending=False)
    return similar_scores.iloc[1:top_n+1].index.tolist()

product_names = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()

def recommend_product_names(product_code, top_n=5):
    recommended_codes = recommend_products(product_code, top_n)
    return [product_names.get(code, code) for code in recommended_codes]

# ------------------ 16. Example Recommendation ------------------
sample_code = similarity_df.columns[0]
print(f"\nTop 5 similar products to '{sample_code}':")
print(recommend_product_names(sample_code))

similarity_df.to_csv('product_similarity_matrix.csv')
