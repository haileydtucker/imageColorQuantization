# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from matplotlib import image as img

# load the image
image_path = 'IMG_1186-EFFECTS.jpg'
image = img.imread(image_path)
if image is None:
    print('********************************************')
    print('*** Unable to load image', image_path)
    print('********************************************\n')
image.shape

from skimage.transform import resize
# Resize to make things faster
image = resize(image, (image.shape[0] // 2, image.shape[1] // 2),
anti_aliasing=True, mode='constant')
# Normalize pixel values between 0 and 1
image = image / image.max()
# get the height, width, and number of color channels (3)
h, w, ch = image.shape
print('Image loaded (', h, 'x', w, ')')
# show the image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
# each pixel is a row with 3 columns (R G B)
pixels = image.reshape((h * w, 3))
print('pixels is', pixels.shape)
pixels[:5]

# ---------------------------------------------------------------------
# Show 3D plot of all the pixels in terms of red, green, and blue
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('RGB')
#ax.axis('equal')
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
data = pixels
subsample = 5
data = data[::subsample]
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data)
plt.show()

# cluster the pixel intensities

k = 13
print('Running Kmeans on list of', pixels.shape[0], 'pixels with k =', k
, '...')
clt = KMeans(n_clusters=k)
clt.fit(pixels)
print('Kmeans complete')
# Assign each pixel to the closest cluster center
labels = clt.predict(pixels)
labels[:5]

print('These are the locations of the cluster centers (R G B)')

clt.cluster_centers_

# Get the list of cluster centers
colors = clt.cluster_centers_
# Count how many pixels have been assigned to each cluster label
counts = {}
for label in labels:
    if not label in counts:
        counts[label] = 1
    else:
        counts[label] += 1
# Show histogram of pixel counts for each cluster
plt.figure()
x = np.arange(k)
plt.bar(x, counts.values(), color=colors, width=1)
plt.xticks(x)
plt.xlabel('Colors')
plt.ylabel('# Pixels')
plt.show()

# Assign each pixel to be the color of its closest cluster center

quant = colors[labels]
print(quant[:5])
# reshape list of RGB values back into an image
img = quant.reshape((h, w, ch))
plt.figure()
plt.imshow(img)
plt.show()
