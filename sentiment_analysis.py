from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained("./model_save")

# Load your trained model
model = BertForSequenceClassification.from_pretrained("./model_save")

# Ensure the model is in evaluation mode
model.eval()


def sentiment_analysis(text):
    # Encode text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # Take the first value of the output as the sentiment score
    sentiment_score = outputs.logits[0][0].item()
    return sentiment_score


# Example usage
text = "The market had a great day with stocks soaring to new highs."
score = sentiment_analysis(text)
print(f"Sentiment score: {score}")
