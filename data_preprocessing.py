import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch


file_path = "dataset_data.csv"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_data(file_path, encoding):
    df = pd.read_csv(file_path, encoding=encoding)
    sentences = df.iloc[:, 1].values

    labels = df.iloc[:, 0].values
    label_dict = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    labels = [label_dict[label] for label in labels]

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    train_inputs, validation_inputs, train_masks, validation_masks = train_test_split(
        input_ids, attention_masks, random_state=2018, test_size=0.1
    )
    train_labels, validation_labels = train_test_split(
        labels, random_state=2018, test_size=0.1
    )

    return (
        train_inputs,
        train_labels,
        train_masks,
        validation_inputs,
        validation_labels,
        validation_masks,
    )


(
    train_inputs,
    train_labels,
    train_masks,
    validation_inputs,
    validation_labels,
    validation_masks,
) = preprocess_data(file_path, "ISO-8859-1")
