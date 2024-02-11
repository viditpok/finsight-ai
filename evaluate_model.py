import torch
from transformers import BertForSequenceClassification, BertTokenizer
from data_preprocessing import preprocess_data
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np
from sklearn.metrics import classification_report
import json

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("./model_save")
tokenizer = BertTokenizer.from_pretrained("./model_save")

# Assuming the 'preprocess_data' function returns the data in the correct format
validation_inputs, validation_labels, validation_masks = preprocess_data(
    "dataset_data.csv", "ISO-8859-1"
)[3:]

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create DataLoader
batch_size = 32
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=batch_size
)


def evaluate_model(model, validation_dataloader):
    model.eval()
    predictions, true_labels = [], []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    return classification_report(true_labels, predictions, output_dict=True)


# Evaluate the model
eval_report = evaluate_model(model, validation_dataloader)

# Print the classification report
print(json.dumps(eval_report, indent=4))

# Optionally, save the report to a file
with open("evaluation_report.json", "w") as f:
    json.dump(eval_report, f)
