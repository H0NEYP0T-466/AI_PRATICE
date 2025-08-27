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


fig4=plt.gcf()  # Get the current figure
plt.close("fig4")

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, 
         color="red",        # line color
         marker="o",         # circle marker
         linestyle="--",     # dashed line
         linewidth=2, 
         markersize=8,
         label="Styled Line")

plt.title("Customized Line Plot")
plt.legend()
plt.show()

fig5=plt.gcf()  # Get the current figure
plt.close("fig5")


plt.plot(x, y, marker="s", color="green")

plt.xlim(0, 6)              # x-axis range
plt.ylim(0, 12)             # y-axis range
plt.xticks([0, 2, 4, 6])    # custom ticks
plt.yticks([0, 5, 10])      

plt.title("Axis Customization Example")
plt.show()

fig6=plt.gcf()  # Get the current figure
plt.close("fig6")

# Method 1: subplot()
plt.subplot(1, 2, 1)   # 1 row, 2 cols, first plot
plt.plot(x, y, color="blue")
plt.title("Plot 1")

plt.subplot(1, 2, 2)   # second plot
plt.plot(y, x, color="orange")
plt.title("Plot 2")

plt.tight_layout()
plt.show()

# Method 2: subplots()
fig, ax = plt.subplots(2, 1, figsize=(6,6))  # 2 rows, 1 col
ax[0].plot(x, y, color="purple")
ax[0].set_title("First Plot")

ax[1].scatter(y, x, color="brown")
ax[1].set_title("Second Plot")

plt.tight_layout()
plt.show()





image = np.random.rand(28, 28)

plt.imshow(image, cmap='gray')  # 'gray' for grayscale
plt.title("Grayscale Image")
plt.axis("off")  # hide axis
plt.show()


rgb_image = np.random.rand(28, 28, 3)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Grayscale")

plt.subplot(1, 2, 2)
plt.imshow(rgb_image)
plt.title("RGB")

plt.show()



# Let's pretend we have 6 images
images = [np.random.rand(28, 28) for _ in range(6)]

plt.figure(figsize=(8, 6))

for i in range(6):
    plt.subplot(2, 3, i+1)  # 2 rows, 3 cols
    plt.imshow(images[i], cmap='gray')
    plt.axis("off")
    plt.title(f"Image {i+1}")

plt.tight_layout()
plt.show()


# Example: classifier predicted digit
pred_label = 3
true_label = 8

plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {pred_label} | Actual: {true_label}", color="red")
plt.axis("off")
plt.show()


cm = np.array([[5, 2, 0],
               [1, 7, 1],
               [0, 2, 9]])

classes = ["Cat", "Dog", "Rabbit"]

plt.imshow(cm, cmap="Blues")
plt.colorbar()

# Add text labels
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


import time

losses = []

plt.ion()  # turn on interactive mode
fig, ax = plt.subplots()

for epoch in range(1, 21):
    loss = np.exp(-epoch/5) + np.random.rand()*0.05  # fake loss
    losses.append(loss)
    
    ax.clear()
    ax.plot(losses, marker="o", color="red")
    ax.set_title(f"Training Loss (Epoch {epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.pause(0.5)  # pause to update
    
plt.ioff()
plt.show()
