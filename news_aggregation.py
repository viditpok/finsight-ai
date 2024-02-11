import requests
from sentiment_analysis import sentiment_analysis
from aggregator import find_company
from pymongo.mongo_client import MongoClient

# Your News API key
API_KEY = "5393befd6d0745cd8d00905c6df9ffde"
BASE_URL = "https://newsapi.org/v2/everything"

# Set up parameters for the news query
params = {
    "q": "finance OR markets",  # Keywords to search for
    "sortBy": "publishedAt",  # Sort by the most recent articles
    "apiKey": API_KEY,
    "language": "en",  # Assuming you want articles in English
}


# Function to fetch news data and analyze sentiment
def fetch_and_analyze_news():
    articles_data = []  # List to store article data and sentiment scores
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raises an HTTPError if the response was an error

        # Parse the JSON response
        articles = response.json().get("articles", [])

        # Process the articles
        for i in range(10):
            article = articles[i]
            company = find_company(article["title"])
            sentiment_score = sentiment_analysis(article["title"])
            article_data = {
                "title": article["title"],
                "source": article["source"]["name"],
                "publishedAt": article["publishedAt"],
                "url": article["url"],
                "content": article["content"],
                "sentiment_score": sentiment_score,
                "company": company
            }
            articles_data.append(article_data)

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return articles_data


# Function to handle MongoDB insertion (to be implemented)
def insert_into_mongodb(articles_data):
    try:
        # MongoDB connection URI
        uri = "mongodb+srv://aabidsq:XHITeTqfHF3v3ya3@datacluster.m435l2t.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri)

        # Access the database
        db = client["news_database"]

        # Access or create the collection
        collection = db["articles"]

        # Insert articles data into the collection
        collection.insert_many(articles_data)

        print("Articles inserted into MongoDB successfully!")

    except Exception as e:
        print(f"Error inserting articles into MongoDB: {e}")
