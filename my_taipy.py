import taipy.gui as gui
import requests
from sentiment_analysis import (
    sentiment_analysis,
)  # Make sure this function returns a sentiment score


# Define the function to fetch news and analyze sentiment
def fetch_and_analyze_news(gui, **kwargs):
    API_KEY = "5393befd6d0745cd8d00905c6df9ffde"
    BASE_URL = "https://newsapi.org/v2/everything"
    params = {
        "q": "finance OR markets",
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
        "language": "en",
    }
    response = requests.get(BASE_URL, params=params)
    articles = response.json().get("articles", [])
    results = "<h3>Fetched News and Their Sentiment Analysis</h3><ul>"
    for article in articles[:5]:  # Limit to first 5 articles for demonstration
        sentiment = sentiment_analysis(
            article["title"]
        )  # Assume this returns a sentiment label
        results += f"<li><strong>Title:</strong> {article['title']}<br><strong>Sentiment:</strong> {sentiment}</li>"
    results += "</ul>"
    gui.get_node("news_results").value = results  # Directly update the GUI node


# Create the GUI using a markdown string
md = """
# Financial News Sentiment Analyzer

<button onclick="fetch_and_analyze_news">Fetch and Analyze News</button>

<div id="news_results"></div>
"""

# Setup GUI
app = gui.Gui(md)

# Bind the action
app.bind_action("fetch_and_analyze_news", fetch_and_analyze_news)

if __name__ == "__main__":
    app.run()
