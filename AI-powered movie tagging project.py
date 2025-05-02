import pandas as pd
import numpy as np
import ast  # Helps parse genres stored as string lists
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("Movies.csv")

# Step 2: Convert genres from string to actual lists
df["Genres"] = df["Genres"].apply(ast.literal_eval)

# Step 3: Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df["Genres"])

# Step 4: Convert encoded genres into DataFrame
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)  # Merge encoded genres back to main dataset

# Step 5: Scale features for clustering
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.iloc[:, 3:])  # Exclude title/description

# Step 6: Apply KMeans Clustering
num_clusters = 3  # Set based on desired number of groups
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Step 7: Evaluate clustering quality (Silhouette Score)
score = silhouette_score(scaled_features, df["Cluster"])
print(f"Silhouette Score: {score:.3f}")

# Step 8: Visualize Clusters
sns.scatterplot(x=df["Sci-Fi"], y=df["Drama"], hue=df["Cluster"], palette="viridis")
plt.title("Movie Clusters Based on Genre")
plt.xlabel("Sci-Fi Score")
plt.ylabel("Drama Score")
plt.show()

# Step 9: Automatic Tagging
df["Generated Tags"] = df["Genres"].apply(lambda x: ", ".join(x) + " Movie")
print(df[["Movie Title", "Generated Tags"]])
