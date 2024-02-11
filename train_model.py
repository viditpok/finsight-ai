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


# Function to preprocess the data
def preprocess_data(file_path, encoding="ISO-8859-1"):
    df = pd.read_csv(file_path, encoding=encoding, header=None, names=["label", "text"])

    # Map labels to -1, 0, 1 for negative, neutral, positive respectively
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
    labels = torch.tensor(
        labels, dtype=torch.float
    )  # Ensure labels are float for regression

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


# Specify the correct path to your dataset
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
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=1
)  # num_labels=1 for regression

# Tell PyTorch to use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Create the learning rate scheduler
total_steps = len(train_inputs) * 4  # Number of epochs times number of batches
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Convert to TensorDataset and create DataLoader
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


# Define the training loop
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


loss_values = []
epochs = 4

for epoch_i in range(0, epochs):
    # Perform one full pass over the training set
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
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

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

# Save the model
model.save_pretrained("./model_save")
tokenizer.save_pretrained("./model_save")

# After saving the model, you can load it using the `from_pretrained` method
# model = BertForSequenceClassification.from_pretrained("./model_save")
