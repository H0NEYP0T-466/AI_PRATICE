from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
images, labels = digits.images[:10], digits.target[:10]
preds = labels.copy()
preds[2] = (preds[2] + 1) % 10  # fake a wrong prediction

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap="gray")
    color = "green" if preds[i] == labels[i] else "red"
    plt.title(f"P:{preds[i]} | A:{labels[i]}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()
