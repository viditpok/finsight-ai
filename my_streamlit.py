import streamlit as st
import pandas as pd
from mongo import client  # Importing the MongoClient connection from mongo.py
from aggregator import autocorrect_Ticker, fixed_sentiment
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Set page config to use a wide layout
st.set_page_config(layout="wide")

# Create three columns
left_spacer, middle_column, right_spacer = st.columns([0.25, 1, 0.25])

# Custom CSS for styling directly in the Streamlit app
def custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@300&display=swap');

        html, body, .stApp {
            background-image: linear-gradient(to bottom, #065535, black);
            font-family: 'Josefin Sans', sans-serif;
            color: white;
        }
        .small-text {
            font-family: 'Josefin Sans', sans-serif;
            font-size: 0.5rem; /* Smaller font size */
            color: white; /* or any color you prefer */
        }
        h1 {
            font-family: 'Josefin Sans', sans-serif;
            text-align: center;
            font-size: 4rem; /* Adjust this value as needed to make the title larger */
            color: white; /* Ensures that the title is white */
        }
        /* Custom style for the paragraph/instructional text */
        p.custom-instruction {
            text-align: center;
            font-family: 'Josefin Sans', sans-serif;
            color: white;
            font-size: 1rem; /* Smaller text */
            margin-bottom: 20px; /* Add some margin at the top */
        }
        /* Styling for the search bar */
        .stTextInput>div>div>input {/
            font-family: 'Josefin Sans', sans-serif;
            font-size: 1.25rem; /* Increase font size */
            border-radius: 50px !important;
            border: 2px solid white !important;
            padding: 10px 12px; /* Increase padding for larger overall size */
            width: 100%;
        }
        .card {
            margin: 10px 0;
            border-radius: 10px;
            background-color: #f0f2f6;
            padding: 15px;
            font-family: 'Josefin Sans', sans-serif;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            color: #333;
        }
        .card-title {
            font-size: 1.25rem;
            font-family: 'Josefin Sans', sans-serif;
            font-weight: bold;
        }
        .card-content {
            margin-top: 10px;
            font-family: 'Josefin Sans', sans-serif;
        }
        .stTextInput > div > div > input, .stButton>button {
            font-family: 'Josefin Sans', sans-serif;
        }
        
        
        </style>
        """, unsafe_allow_html=True)


custom_css()



with middle_column:
    # Streamlit app layout
    st.markdown('<h1>FINSIGHT AI</h1>', unsafe_allow_html=True)

    # Centered instructional text below the title
    st.markdown('<p class="custom-instruction">Enter a company name for analysis:</p>', unsafe_allow_html=True)
    # User input for company name
    user_input_company_name = st.text_input("", placeholder="Enter a company name...")

# Function to retrieve articles from MongoDB based on ticker symbol
def get_articles_by_ticker(ticker_symbol):
    db = client['news_database']
    collection = db['articles']
    query = {"company": ticker_symbol.upper()}
    articles = list(collection.find(query))
    return articles

# Function to calculate stock direction prediction based on sentiment scores
def predict_stock_direction(articles):
    if not articles:
        return "No data for prediction."
    fixed_sentiment(articles)
    avg_sentiment_score = sum(article['sentiment_score'] for article in articles) / len(articles)
    return "Stock likely to go up." if avg_sentiment_score > 0 else "Stock likely to go down."

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_arrow(avg_sentiment_score):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Determine the color, direction, and label based on avg_sentiment_score
    if avg_sentiment_score > 0:
        color = 'green'
        arrowstyle = 'simple,head_width=2,head_length=2,tail_width=1'
        direction = 'upward'
        dx, dy = 0, 0.3
        x, y = 0.2, 0.5  # Arrow position
    elif avg_sentiment_score < 0:
        color = 'red'
        arrowstyle = 'simple,head_width=2,head_length=2,tail_width=1'
        direction = 'downward'
        dx, dy = 0, -0.3
        x, y = 0.2, 0.5  # Arrow position
    else:
        color = 'grey'
        arrowstyle = 'simple,head_width=1,head_length=1,tail_width=0.5'
        direction = 'stable'
        dx, dy = 0.3, 0
        x, y = 0.2, 0.5  # Arrow position

    # Create the arrow
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                            arrowstyle=arrowstyle, color=color, mutation_scale=10)
    ax.add_patch(arrow)

    # Add direction text above the arrow
    ax.text(0.6, 0.7, direction, ha='center', va='center', fontsize=20, color=color)

    # Add sentiment score text below the arrow on the same line
    ax.text(0.5 + dx, 0.5 + dy, f"  {avg_sentiment_score:.1f}", ha='left', va='center', fontsize=6)

    # Remove the axis for a cleaner look
    plt.axis('off')

    return fig



if user_input_company_name:
    # Correct the company name to its ticker symbol
    ticker_symbol = autocorrect_Ticker(user_input_company_name)
    
    if ticker_symbol:
        # Retrieve related articles from MongoDB
        articles = get_articles_by_ticker(ticker_symbol)
        articles_df = pd.DataFrame(articles)
        
        # Create two columns for the layout
        left_column, right_column = st.columns(2)
        
        with left_column:
            if not articles_df.empty:
                # Display the ticker symbol and number of articles found with custom styling
                st.markdown(f"""
                    <div style='text-align: center; color: white; font-family: "Josefin Sans", sans-serif; margin-top: 20px; font-size: 0.75rem;'>
                        Ticker symbol: <span style='color: gold;'>{ticker_symbol}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='text-align: center; font-family: "Josefin Sans", sans-serif; color: white; font-size: 0.75rem; margin-bottom: 20px;'>
                        Articles Found: <span style='color: gold;'>{len(articles_df)}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Loop through each article and display its title and sentiment score
                for index, row in articles_df.iterrows():
                    sentiment_color = "red" if row['sentiment_score'] < 0 else "green"
                    # Use an <a> tag for the title to make it clickable
                    st.markdown(f"""
                        <div class="card" style="cursor: pointer;" onclick="window.open('{row['url']}', '_blank');">
                            <div class="card-title">
                                <a href="{row['url']}" target="_blank" style="text-decoration: none; color: DodgerBlue;">
                                    {row['title']}
                                </a>
                            </div>
                            <div class="card-content" style='color: {sentiment_color};'>Sentiment Score: {row['sentiment_score']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"No articles found for the ticker symbol '{ticker_symbol}'.")

        with right_column:
            # Only attempt to display the stock direction prediction if articles were found
            if not articles_df.empty:
                # Compute the average sentiment score
                avg_sentiment_score = sum(article['sentiment_score'] for article in articles) / len(articles)
                
                # Display the stock direction prediction
                stock_direction_prediction = predict_stock_direction(articles)
                
                st.markdown("""
                    <div style="font-family: 'Josefin Sans', sans-serif; text-align: center; margin-top: 20px; font-size: 24px;">Stock Direction Prediction</div>
                """, unsafe_allow_html=True)
                
                st.write(stock_direction_prediction)
                
                # Display the arrow plot
                fig = plot_arrow(avg_sentiment_score)
                st.pyplot(fig)
            else:
                st.error(f"No articles found for the ticker symbol '{ticker_symbol}'.")
    else:
        st.error("Could not find a ticker symbol for the given company name.")
