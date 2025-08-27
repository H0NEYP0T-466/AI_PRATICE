from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
labels = digits.target

plt.hist(labels, bins=len(set(labels)), edgecolor="black")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Class Distribution in Digits Dataset")
plt.show()
