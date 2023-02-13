import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector

# Define the size of the image
x_size, y_size = 100, 100

# Create an empty image with all pixels set to 0 (black)
img = np.zeros((x_size, y_size))

# Plot the image
fig, ax = plt.subplots()
plt.imshow(img, cmap="gray")

# Define a function to be called when a lasso is drawn
def onselect(verts):
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        img[int(y1), int(x1)] = 1
        img[int(y2), int(x2)] = 1
        # Compute the line between the two points
        x = np.linspace(x1, x2, int(np.abs(x2 - x1)))
        y = np.linspace(y1, y2, int(np.abs(y2 - y1)))
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        # Set the pixels along the line to white
        for i in range(len(x)):
            img[y[i], x[i]] = 1
    plt.imshow(img, cmap="gray")
    fig.canvas.draw()


# Enable lasso selection
ls = LassoSelector(ax, onselect)

plt.show()
