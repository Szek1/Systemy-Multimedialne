import matplotlib.pyplot as plt
import numpy as np
import cv2


def nearest_neighbor(img, scale):
    height = img.shape[0]
    width = img.shape[1]

    new_width = int(width * scale)
    new_height = int(height * scale)

    if (len(img.shape) < 3):
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        channel = img.shape[2]
        new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)
    X = np.round(np.linspace(0, height - 1, new_height)).astype(int)
    Y = np.round(np.linspace(0, width - 1, new_width)).astype(int)

    for i in range(new_height):
        for j in range(new_width):
            new_img[i, j] = img[X[i], Y[j]]
    return new_img


def bilinear_interpolation(img, scale):
    height = img.shape[0]
    width = img.shape[1]
    new_height = int(height * scale)
    new_width = int(width * scale)
    if (len(img.shape) < 3):
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        channel = img.shape[2]
        new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)
    X = np.linspace(0, height - 1, new_height)
    Y = np.linspace(0, width - 1, new_width)

    for i in range(new_height):
        for j in range(new_width):
            Px = X[i]
            Py = Y[j]

            x1 = int(Px)
            x2 = min(x1 + 1, height - 1)
            y1 = int(Py)
            y2 = min(y1 + 1, width - 1)

            x = Px - x1
            y = Py - y1

            q11 = img[x1, y1]
            q12 = img[x1, y2]
            q21 = img[x2, y1]
            q22 = img[x2, y2]

            new_img[i, j] = q11 * (1 - x) * (1 - y) + q21 * x * (1 - y) + q12 * (1 - x) * y + q22 * x * y

    return new_img


def mean(img, scale):
    height = img.shape[0]
    width = img.shape[1]

    new_height = np.ceil(height / scale).astype(int)
    new_width = np.ceil(width / scale).astype(int)

    X = np.linspace(0, height - 1, new_height).astype(int)
    Y = np.linspace(0, width - 1, new_width).astype(int)

    if (len(img.shape) < 3):
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        channel = img.shape[2]
        new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            ix = np.clip(np.round(X[i] + np.arange(-3, 4)), 0, height - 1).astype(int)
            iy = np.clip(np.round(Y[j] + np.arange(-3, 4)), 0, width - 1).astype(int)

            if (len(img.shape) < 3):
                new_img[i, j] = np.mean(img[ix, iy])
            else:
                for k in range(channel):
                    new_img[i, j, k] = np.mean(img[ix, iy, k])
    return new_img


def median(img, scale):
    height = img.shape[0]
    width = img.shape[1]

    new_height = np.ceil(height / scale).astype(int)
    new_width = np.ceil(width / scale).astype(int)

    X = np.linspace(0, height - 1, new_height).astype(int)
    Y = np.linspace(0, width - 1, new_width).astype(int)

    if (len(img.shape) < 3):
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        channel = img.shape[2]
        new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            ix = np.clip(np.round(X[i] + np.arange(-3, 4)), 0, height - 1).astype(int)
            iy = np.clip(np.round(Y[j] + np.arange(-3, 4)), 0, width - 1).astype(int)
            if (len(img.shape) < 3):
                new_img[i, j] = np.median(img[ix, iy])
            else:
                for k in range(channel):
                    new_img[i, j, k] = np.median(img[ix, iy, k])
    return new_img


def weighted_mean(img, scale):
    height = img.shape[0]
    width = img.shape[1]

    new_height = np.ceil(height / scale).astype(int)
    new_width = np.ceil(width / scale).astype(int)

    X = np.linspace(0, height - 1, new_height).astype(int)
    Y = np.linspace(0, width - 1, new_width).astype(int)

    if (len(img.shape) < 3):
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        channel = img.shape[2]
        new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            ix = np.clip(np.round(X[i] + np.arange(-3, 4)), 0, height - 1).astype(int)
            iy = np.clip(np.round(Y[j] + np.arange(-3, 4)), 0, width - 1).astype(int)
            weights = [5, 10, 15, 20, 15, 10, 5]

            ixx = (np.sum(np.multiply(ix, weights)) / np.sum(weights)).astype(int)
            iyy = (np.sum(np.multiply(iy, weights)) / np.sum(weights)).astype(int)
            new_img[i, j] = img[ixx, iyy]

    return new_img


img1 = plt.imread('0001.jpg')
fragment = img1[10:70, 5:80].copy()

# 1

# plt.subplot(1, 2, 1)
# plt.title("Oryginalny obraz")
# plt.imshow(img1)
#
# plt.subplot(1, 2, 2)
# plt.title("Fragment")
# plt.imshow(fragment)
#
# plt.subplot(2, 2, 3)
# plt.title("Najbliższy sąsiad")
# plt.imshow(nearest_neighbor(fragment, 5))
#
# plt.subplot(2, 2, 4)
# plt.title("Interpolacja dwuliniowa")
# plt.imshow(bilinear_interpolation(fragment, 5))
# plt.show()

# plt.subplot(3, 2, 1)
# plt.title("Najbliższy sąsiad 5%")
# plt.imshow(nearest_neighbor(fragment, 0.95))
#
# plt.subplot(3, 2, 2)
# plt.title("Interpolacja dwuliniowa 5%")
# plt.imshow(bilinear_interpolation(fragment, 1.05))
#
# plt.subplot(3, 2, 3)
# plt.title("Najbliższy sąsiad 20%")
# plt.imshow(nearest_neighbor(fragment, 1.2))
#
# plt.subplot(3, 2, 4)
# plt.title("Interpolacja dwuliniowa 20%")
# plt.imshow(bilinear_interpolation(fragment, 1.2))
#
# plt.subplot(3, 2, 5)
# plt.title("Najbliższy sąsiad 80%")
# plt.imshow(nearest_neighbor(fragment, 1.8))
#
# plt.subplot(3, 2, 6)
# plt.title("Interpolacja dwuliniowa 80%")
# plt.imshow(bilinear_interpolation(fragment, 1.8))
#
# plt.show()

# 2

img2 = plt.imread('0007.jpg')
fragment2 = img2[0:2000, 2800:4500].copy()

# plt.subplot(1,2,1)
# plt.title("Oryginalny obraz")
# plt.imshow(img2)
# plt.subplot(1,2,2)
# plt.title("Fragment")
# plt.imshow(fragment2)
# plt.show()

# plt.subplot(2,2,1)
# plt.title("Fragment")
# plt.imshow(fragment2)
# plt.subplot(2,2,2)
# plt.title("10%")
# plt.imshow(mean(fragment2,1.1))
# plt.subplot(2,2,3)
# plt.title("100")
# plt.imshow(mean(fragment2,2))
# plt.subplot(2,2,4)
# plt.title("300%")
# plt.imshow(mean(fragment2,4))
# plt.show()

# plt.subplot(2, 2, 1)
# plt.title("Fragment")
# plt.imshow(fragment2)
# plt.subplot(2, 2, 2)
# plt.title("10%")
# plt.imshow(median(fragment2, 1.1))
# plt.subplot(2, 2, 3)
# plt.title("100")
# plt.imshow(median(fragment2, 2))
# plt.subplot(2, 2, 4)
# plt.title("300%")
# plt.imshow(median(fragment2, 4))
# plt.show()
#
# plt.subplot(2, 2, 1)
# plt.title("Fragment")
# plt.imshow(fragment2)
# plt.subplot(2, 2, 2)
# plt.title("10%")
# plt.imshow(weighted_mean(fragment2, 1.1))
# plt.subplot(2, 2, 3)
# plt.title("100")
# plt.imshow(weighted_mean(fragment2, 2))
# plt.subplot(2, 2, 4)
# plt.title("300%")
# plt.imshow(weighted_mean(fragment2, 4))
# plt.show()

# plt.subplot(2,2,1)
# plt.title("Fragment")
# plt.imshow(fragment2)
# plt.subplot(2,2,2)
# plt.title("Średnia")
# plt.imshow(mean(fragment2,15))
# plt.subplot(2,2,3)
# plt.title("Mediana")
# plt.imshow(median(fragment2,15))
# plt.subplot(2,2,4)
# plt.title("Średnia ważona")
# plt.imshow(weighted_mean(fragment2,15))
# plt.show()


img3 = cv2.imread('0008.tif', cv2.IMREAD_GRAYSCALE)
img3 = img3[260:500, 20:480]
edges = cv2.Canny(img3, 100, 200).copy()

# plt.subplot(3, 1, 1)
# plt.title("Oryginalne")
# plt.imshow(edges,cmap='gray')
# plt.subplot(3, 1, 2)
# plt.title("Najblizszy sasiad")
# plt.imshow(nearest_neighbor(edges, 10),cmap='gray')
# plt.subplot(3, 1, 3)
# plt.title("Interpolacja dwuliniowa")
# plt.imshow(bilinear_interpolation(edges, 10),cmap='gray')
# plt.show()

plt.subplot(2,2,1)
plt.title("Oryginalne")
plt.imshow(edges,cmap='gray')

plt.subplot(2,2,2)
plt.title("Średnia")
plt.imshow(mean(edges,3),cmap='gray')

plt.subplot(2,2,3)
plt.title("Mediana")
plt.imshow(median(edges,3),cmap='gray')

plt.subplot(2,2,4)
plt.title("Średnia ważona")
plt.imshow(weighted_mean(edges,3),cmap='gray')
plt.show()