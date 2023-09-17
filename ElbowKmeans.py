#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy pandas matplotlib scikit-learn')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from CSV file
data = pd.read_csv("RFMdatanew.csv")

# Extract the features (normalized recency and normalized amount) for clustering
features = data.iloc[:, 1:3].values

# Calculate within-cluster sum of squares (WCSS) for different values of k
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




