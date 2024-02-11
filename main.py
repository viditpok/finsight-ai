import news_aggregation

articles_data = news_aggregation.fetch_and_analyze_news()
news_aggregation.insert_into_mongodb(articles_data)
