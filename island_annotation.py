from skimage.segmentation import active_contour
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
import numpy as np
import glob
import os

df = pd.DataFrame(columns=[str(x) for x in range(800)])

image_paths = glob.glob('.\\images\\*.jpg')

if not os.path.exists("images_annotated"):
    os.mkdir("images_annotated")
for image_path in image_paths:
    print(image_path)
    image = imread(image_path)
    
    if image.shape[0] > image.shape[1]:
        image = np.pad(image, ((0,0), ((image.shape[0] - image.shape[1])//2, (image.shape[0] - image.shape[1])//2)), 'constant', constant_values=(0,0))
    else:
        image = np.pad(image, (((image.shape[1] - image.shape[0])//2, (image.shape[1] - image.shape[0])//2), (0,0)), 'constant', constant_values=(0,0))

    radius = min(image.shape[0], image.shape[1]) // 2

    s = np.linspace(0, 2*np.pi, 400)
    r = image.shape[0]//2 + radius*np.sin(s)
    c = image.shape[1]//2 + radius*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(image, 3, preserve_range=False), init, alpha=0.015, beta=0.01, gamma=0.01)

    df.loc[len(df.index)] = snake.flatten() 

    '''
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)        # Show the initial snake
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)       # Show the final snake
    ax.set_xticks([]), ax.set_yticks([])    
    ax.axis([0, image.shape[1], image.shape[0], 0])
    plt.show()
    
    plt.savefig(os.path.join('.\\images_annotated\\', image_path.split('\\')[-1]))
    '''
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])  
    ax.axis([0, image.shape[1], image.shape[0], 0])
    plt.savefig(".\\images_annotated\\{0}_annotated.jpg".format(image_path.split('\\')[-1].split(".")[0]))
    plt.close()

# Add the path column as label and save the snakes as a dataframe
df['image_path'] = image_paths
df.to_csv("island_annotation.csv", index=False)

# Extract the feature matrix and calculate the mean features.
data = df.drop(columns=['image_path']).values.transpose()
mean_features = data.mean(axis=1)

# Reshape the features to get the snake
mean_snake = mean_features.reshape((-1,2))

# Save the mean snake as an image
im = np.zeros((700, 700))

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(mean_snake[:, 1], mean_snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])
plt.savefig("mean_island.jpg")

# Put mean snake on an unseen island
im = imread('fn01.jpg')

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(mean_snake[:, 1], mean_snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])
plt.savefig("fn01_annot.jpg")