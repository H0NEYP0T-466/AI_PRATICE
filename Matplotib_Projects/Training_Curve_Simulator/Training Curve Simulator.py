import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 21)
train_acc = np.linspace(0.5, 0.95, 20) + np.random.rand(20)*0.02
val_acc   = np.linspace(0.45, 0.90, 20) + np.random.rand(20)*0.03

plt.plot(epochs, train_acc, label="Training Accuracy", marker="o")
plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
