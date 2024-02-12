import json
import matplotlib.pyplot as plt
import numpy as np


with open("evaluation_report.json", "r") as f:
    eval_report = json.load(f)


classes = list(eval_report.keys())[:-3]
precision = [eval_report[cls]["precision"] for cls in classes]
recall = [eval_report[cls]["recall"] for cls in classes]
f1_score = [eval_report[cls]["f1-score"] for cls in classes]


ind = np.arange(len(classes))
width = 0.25


plt.figure(figsize=(12, 6))
plt.bar(ind, precision, width, label="Precision")
plt.bar(ind + width, recall, width, label="Recall")
plt.bar(ind + 2 * width, f1_score, width, label="F1 Score")

plt.xlabel("Classes")
plt.ylabel("Scores")
plt.title("Classification Metrics by Class")

plt.xticks(ind + width, classes)
plt.legend(loc="best")
plt.show()
