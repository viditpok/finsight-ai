import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime


def preprocess_data(file_path, encoding="ISO-8859-1"):
    df = pd.read_csv(file_path, encoding=encoding, header=None, names=["label", "text"])

    df["label"] = df["label"].map({"negative": -1, "neutral": 0, "positive": 1})

    sentences = df["text"].values
    labels = df["label"].values

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
    labels = torch.tensor(labels, dtype=torch.float)

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


file_path = "dataset_data.csv"
(
    train_inputs,
    train_labels,
    train_masks,
    validation_inputs,
    validation_labels,
    validation_masks,
) = preprocess_data(file_path, "ISO-8859-1")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)


total_steps = len(train_inputs) * 4
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
validation_dataset = TensorDataset(
    validation_inputs, validation_masks, validation_labels
)

train_dataloader = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=32
)
validation_dataloader = DataLoader(
    validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=32
)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


loss_values = []
epochs = 4

for epoch_i in range(0, epochs):

    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)

            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")


model.save_pretrained("./model_save")
tokenizer.save_pretrained("./model_save")
