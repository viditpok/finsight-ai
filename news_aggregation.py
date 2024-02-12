import requests
from sentiment_analysis import sentiment_analysis
from aggregator import find_company
from pymongo.mongo_client import MongoClient


API_KEY = "5393befd6d0745cd8d00905c6df9ffde"
BASE_URL = "https://newsapi.org/v2/everything"


params = {
    "q": "finance OR markets",
    "sortBy": "publishedAt",
    "apiKey": API_KEY,
    "language": "en",
}


def fetch_and_analyze_news():
    articles_data = []
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        articles = response.json().get("articles", [])

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
                "company": company,
            }
            articles_data.append(article_data)

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return articles_data


def insert_into_mongodb(articles_data):
    try:

        uri = "mongodb+srv://aabidsq:XHITeTqfHF3v3ya3@datacluster.m435l2t.mongodb.net/?retryWrites=true&w=majority"

        client = MongoClient(uri)

        db = client["news_database"]

        collection = db["articles"]

        collection.insert_many(articles_data)

        print("Articles inserted into MongoDB successfully!")

    except Exception as e:
        print(f"Error inserting articles into MongoDB: {e}")
