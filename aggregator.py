import datetime
import math
import openai
from openai import OpenAI
import os


openai.api_key = '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'

def find_company(article):
    client = OpenAI(api_key='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')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Can you help me determine what ticker symbol "
                "abbreviation this title talks about?"
                "Please only type the symbol in all caps so that our"
                "code can use your response. \n" + article,
            }
        ],
    )
    return response.choices[0].message.content


def autocorrect_Ticker(company_name):
    name = (
        openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": "Given a potentially misspelled company name"
                    + company_name
                    + "find the correct spelling. When you give your response, give your response only in company name in all caps and nothing else.",
                }
            ],
        )
        .choices[0]
        .message.content
    )
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "Given a company name"
                + name
                + "find its tick symbol. When you give your response, give your response only in symbol in all caps and nothing else.",
            }
        ],
    )
    return response.choices[0].message.content


def fixed_sentiment(dicti):
    for item in dicti:
        date_time_obj = datetime.datetime.strptime(
            item["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)

        now = datetime.datetime.now(datetime.timezone.utc)

        time_diff = now - date_time_obj

        days_diff = time_diff.total_seconds() / (60 * 60 * 24)
        days_diff = max(1, days_diff)
        
        item["sentiment_score"] /= (math.log2(days_diff + 1) + 1e-3)
