import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 1. Generate synthetic data (you can replace with real dataset)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

# 2. Scale data (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Add labels to dataframe
df['Cluster'] = labels

# 5. Plot clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()
plt.show()
