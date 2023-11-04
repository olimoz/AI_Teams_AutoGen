
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
import plotly.express as px
import re

# Path to the excel file
file_path = "/home/oliver/Documents/LangChain/ProductDevelopment/MoreClusters/tools_data.xlsx"

# Read the excel file
df = pd.read_excel(file_path)

# Extract the 'Task Name' column
task_names = df['Task Name']

# Initialize the SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Convert the 'Task Name' into embeddings
embeddings = model.encode(task_names)

# Number of clusters
num_clusters = 25

# Initialize the KMeans model
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the embeddings
kmeans.fit(embeddings)

# Get the cluster assignments
cluster_assignments = kmeans.labels_

# Initialize a list to store the cluster summaries
cluster_summaries = []

# For each cluster
for i in range(num_clusters):
    # Get the task names in the cluster
    tasks_in_cluster = task_names[cluster_assignments == i]
    
    # Split the task names into words and count the frequency of each word
    word_counts = Counter(re.findall(r'\w+', ' '.join(tasks_in_cluster)))
    
    # Get the 5 most common words
    most_common_words = [word for word, _ in word_counts.most_common(5)]
    
    # Join the most common words into a summary and add it to the list of cluster summaries
    cluster_summaries.append(', '.join(most_common_words))

# Add the 'Cluster Assignment' and 'Cluster Summary' columns to the dataframe
df['Cluster Assignment'] = cluster_assignments
df['Cluster Summary'] = [cluster_summaries[i] for i in cluster_assignments]

# Path to the output file
output_file_path = "/home/oliver/Documents/LangChain/ProductDevelopment/MoreClusters/tools_data_clustered.xlsx"

# Save the dataframe to an excel file
df.to_excel(output_file_path, index=False)

# Define the file path
file_path = "/home/oliver/Documents/LangChain/ProductDevelopment/MoreClusters/tools_data_clustered.xlsx"

# Use pandas to read the excel file
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")

# Create a TreeMap
fig = px.treemap(df, path=['Cluster Name'], values='Cluster')

# Save the TreeMap as a jpeg
fig.write_image("/home/oliver/Documents/LangChain/ProductDevelopment/MoreClusters/cluster_treemap.jpeg")

#################
# ERROR
#################
# Code does not complete, GPT-4's 8k token limit exceeded by AutoGen on multiple attempts.
# Project does not execute with GPT-3.5

