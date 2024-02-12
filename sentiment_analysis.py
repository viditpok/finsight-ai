from transformers import BertTokenizer, BertForSequenceClassification
import torch


tokenizer = BertTokenizer.from_pretrained("./model_save")


model = BertForSequenceClassification.from_pretrained("./model_save")


model.eval()


def sentiment_analysis(text):

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    sentiment_score = outputs.logits[0][0].item()
    return sentiment_score


text = "The market had a great day with stocks soaring to new highs."
score = sentiment_analysis(text)
print(f"Sentiment score: {score}")
