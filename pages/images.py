import streamlit as st
import requests
import os
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Define your Unsplash API access key
# UNSPLASH_ACCESS_KEY = "W7iXJCmDYjNzxmqQkA9Envhh_2fvWo7s5uVcBpz83RM"
UNSPLASH_ACCESS_KEY = st.secrets["images_api_key"]


def fetch_images(query, count=20):
    """Fetch images based on the query from the Unsplash API."""
    url = f"https://api.unsplash.com/search/photos/?query={query}&client_id={UNSPLASH_ACCESS_KEY}&per_page={count}"
    response = requests.get(url)
    data = response.json()
    return [photo['urls']['regular'] for photo in data['results']], [photo.get('description') for photo in data['results']]

def download_images(urls, folder_name='images'):
    """Download images from URLs to a specified folder."""
    os.makedirs(folder_name, exist_ok=True) # Create the folder if it doesn't exist 
    for i, url in enumerate(urls):  # Loop through each URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(f"{folder_name}/image_{i}.jpg")

def cluster_images(image_folder, descriptions, num_clusters=3, use_text_clustering=False, resize_dims=(100, 100)):
    """Cluster images based on their content or descriptions."""
    image_files = os.listdir(image_folder)
    valid_descriptions = [desc for desc in descriptions if desc is not None]    
    images = []
    
    # Loop through each image file and its description
    for image_file, desc in zip(image_files, descriptions): 
        if desc is None: # Skip images with no description
            continue
        img = Image.open(os.path.join(image_folder, image_file)) # Open the image file
        img = img.resize(resize_dims)
        img_array = np.array(img)
        images.append(img_array.flatten())
    
    if use_text_clustering: 
        # Text Clustering
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)   # Create a TfidfVectorizer object for text feature extraction
        text_features = vectorizer.fit_transform(valid_descriptions)         # Extract text features from valid descriptions using TfidfVectorizer
        km = KMeans(n_clusters=num_clusters)
        km.fit(text_features)
        return [file for file, desc in zip(image_files, descriptions) if desc is not None], None, km.labels_
    else:
        # Image Content Clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(images)
        return [file for file, desc in zip(image_files, descriptions) if desc is not None], kmeans.labels_, None

# Streamlit UI
st.title("Image Clustering from Unsplash")
st.write("Enter a search query to fetch and cluster images based on their content or text descriptions.")

query = st.text_input("Enter search query:", placeholder="e.g., headphones, cup, bag", help="This field is required.")
num_clusters = st.slider("Number of clusters:", 2, 10, 3)
use_text_clustering = st.checkbox("Cluster by Text Description")

if query: # Check if query is not empty
    if st.button("Scrape and Cluster Images"):
        # Fetch images and descriptions based on the query
        image_urls, descriptions = fetch_images(query)
        # Download the fetched images
        download_images(image_urls)
        # Cluster the images based on their content or descriptions
        image_files, image_labels, text_labels = cluster_images('images', descriptions, num_clusters, use_text_clustering)
        
        if image_labels is not None:
            st.write("Cluster labels (Image Content):")
            st.write(image_labels)
        if text_labels is not None:
            st.write("Cluster labels (Text Description):")
            st.write(text_labels)

        st.write("Images, labels, and descriptions:")
        if image_labels is not None:
            # Display images with their corresponding image labels and descriptions
            for image_file, image_label, description in zip(image_files, image_labels, descriptions):
                st.image(f'images/{image_file}', caption=f'Image Cluster: {image_label}', use_column_width=True)
                if description:
                    st.write(description)
                else:
                    st.write("No description available.")
                st.write("---")
                
        elif text_labels is not None:
            # Display images with their corresponding text labels and descriptions
            for image_file, text_label, description in zip(image_files, text_labels, descriptions):
                st.image(f'images/{image_file}', caption=f'Text Cluster: {text_label}', use_column_width=True)
                if description:
                    st.write(description)
                else:
                    st.write("No description available.")
                st.write("---")
        else:
            # Display images with their descriptions
            for image_file, description in zip(image_files, descriptions):
                st.image(f'images/{image_file}', use_column_width=True)
                if description:
                    st.write(description)
                else:
                    st.write("No description available.")
                st.write("---")
                
else: 
    st.warning("Please enter a search query to fetch images.")

