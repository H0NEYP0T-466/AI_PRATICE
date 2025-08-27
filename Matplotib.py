import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 4, 9, 16, 25]
y3 = [1, 8, 27, 64, 125]

plt.plot(x, y1, label="y = 2x")     # First line
plt.plot(x, y2, label="y = x^2")    # Second line
plt.plot(x, y3, label="y = x^3")    # Third line

plt.title("Legend Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.legend()   
plt.show()
fig1=plt.gcf()  # Get the current figure
#plt.savefig("legend_example.png")  # Save the figure as a PNG file

plt.close("fig1")

epochs = [1, 2, 3, 4, 5]
train_loss = [0.9, 0.7, 0.5, 0.4, 0.3]
val_loss = [1.0, 0.8, 0.6, 0.55, 0.5]

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


fig2 = plt.gcf()  # Get the current figure
plt.close("fig2")
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, c="blue", marker="o", label="Data Points")
plt.xlabel("X Feature")
plt.ylabel("Y Feature")
plt.title("Scatter Plot Example")
plt.legend()
plt.show()


fig3=plt.gcf()  # Get the current figure
plt.close("fig3")
data = np.random.randn(1000)  

plt.hist(data, bins=30, color="purple", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.show()
