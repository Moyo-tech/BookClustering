import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import datetime
import os
import json

# Initialization
st.set_page_config(page_title="Web Page Clusters", layout="wide")
st.title("News Article Clusters")

# Retrieve the API key from Streamlit's secrets management for added security
# api_key = "989cc16af9604d3eadd107badc2388c4"
api_key = st.secrets["news_api_key"]
news_api_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

# Allow the user to select the number of clusters from the sidebar
n_clusters = st.selectbox("Select number of clusters", range(2, 9), index=3)

def fetch_news_articles(url):
    """Fetch articles from the provided URL."""
    try:
      cache_file = "news_cache.json"
      current_date = datetime.datetime.now().strftime("%Y-%m-%d")

      # Check if cache file exists and if the cached data is up-to-date
      if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
          cache_data = json.load(f)
          if cache_data.get("date") == current_date:
            return cache_data["articles"]

      # If cache is not available or outdated, fetch articles from the API
      response = requests.get(url)
      articles = []

      # If the API response is successful, extract the articles and update the cache
      if response.status_code == 200:
        articles = response.json().get("articles", [])
        with open(cache_file, "w") as f:
          json.dump({"date": current_date, "articles": articles}, f)

      return articles
    except requests.RequestException as e:
        st.error(f"Error fetching articles: {e}")  # Handle any request errors gracefully
        return []

def cluster_articles(articles):
    """Cluster articles using KMeans based on their content."""
    contents = [article["description"] or article["content"] for article in articles if article["description"] or article["content"]]
    if not contents: # If no content is available, show a warning and return an empty list
        st.warning("No content available for clustering.") 
        return []

    vectorizer = TfidfVectorizer()  # Initialize a TfidfVectorizer
    X = vectorizer.fit_transform(contents)  # Fit and transform the vectorizer on the content
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)   # Initialize a KMeans model
    kmeans.fit(X) # Fit the model on the vectorized content
    return kmeans.labels_   

def display_clusters(articles, labels):
    """Display the clustered articles."""
    for i in range(n_clusters):
        cluster_articles = [article for article, label in zip(articles, labels) if label == i] # Get articles for each cluster
        if cluster_articles: # If articles are available for the cluster, display them
            st.subheader(f"Cluster {i + 1}")  
            for article in cluster_articles: # Display each article in the cluster
                st.markdown(f"* [{article['title']}]({article['url']})")  


def main():  
    articles = fetch_news_articles(news_api_url) # Fetch articles
    if articles: # If articles are available, cluster and display them
        labels = cluster_articles(articles)
        display_clusters(articles, labels)

if __name__ == "__main__":
    main()
 
