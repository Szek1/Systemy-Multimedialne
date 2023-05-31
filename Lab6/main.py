import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def koder_RLE(img):
    img = img.astype(int)
    x = np.array([len(img.shape)])
    x = np.concatenate([x, img.shape])
    shape = x[1:int(x[0] + 1)]
    retval = []
    cnt = 0
    img = img.flatten()
    for i in range(0, len(img)):
        if (len(img) - 1 != i and img[i] == img[i + 1]):
            cnt += 1
        else:
            retval.append(cnt + 1)
            retval.append(img[i])
            cnt = 0
    return retval, shape


def dekoder_RLE(img):
    img, shape = img
    retval = []
    for i in range(0, len(img), 2):
        retval += [img[i + 1]] * img[i]
    retval = np.reshape(retval, shape)
    return retval


def koder_byte_run(img):
    img = img.astype(int)
    x = np.array([len(img.shape)])
    x = np.concatenate([x, img.shape])
    shape = x[1:int(x[0] + 1)]
    img = img.flatten()
    retval = []
    cnt = 0
    # print(f"img len = {len(img)} ")
    for i in range(0, len(img)):
        if (len(img) - 1 != i and img[i] == img[i + 1]):
            cnt += 1
        else:
            if (cnt != 0):
                retval.append(-cnt)
            retval.append(img[i])
            cnt = 0
    return retval, shape


def dekoder_byte_run(img):
    img, shape = img
    retval = []
    for i in range(0, len(img)):
        if (img[i] < 0):
            retval += ((img[i] * -1)) * [img[i+1]]
        else:
            retval.append(img[i])
    retval = np.reshape(retval,shape)
    return retval


t1 = np.array([1, 1, 1, 2, 3, 4, 4, 1, 2, 3, 4])
t2 = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1])
t3 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
t4 = np.array([5, 1, 5, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5])
t5 = np.array([[4, 4, 2, 1, 6, 6, 6], [5, 1, 2, 2, 3, 3, 9], [6, 3, 1, 7, 7, 3, 2]])

# Przypadki testowe
#
test = [t1, t2, t3, t4, t5]
# print("RLE")
# for i in test:
#     cd = koder_RLE(i)
#     print(f"Koder RLE = {cd}")
#     print(f"Dekoder RLE = {dekoder_RLE(cd)}")
#     print(f"Oryginalne dane = {i}\n")

# print("Byte Run")
# for i in test:
#     cd = koder_byte_run(i)
#     print(f"Koder Byte Run = {cd}")
#     print(f"Dekoder Byte Run = {dekoder_byte_run(cd)}")
#     print(f"Oryginalne dane = {i}\n")

imgRys = cv.imread('rysunekTechniczny.jpg')
imgDoc = cv.imread('dokument.png')
imgColor = plt.imread('kolorowe.jpg')
imgColor = imgColor.astype(int)
imgRys = imgRys.astype(int)
imgDoc = imgDoc.astype(int)


images = [imgRys,imgDoc,imgColor]
imgName = ["Rysunek techniczny", "Dokument", "Kolorowe zdjęcie"]

for i,x in zip(images,imgName):
    cdR1 = koder_RLE(i)
    dcR1 = dekoder_RLE(cdR1)
    size = get_size(i)
    sizeR11 = get_size(cdR1)
    sizeR12 = get_size(dcR1)

    cdB1 = koder_byte_run(i)
    dcB1 = dekoder_byte_run(cdB1)
    sizeB11 = get_size(cdB1)
    sizeB12 = get_size(dcB1)

    print("\tRLE")
    print(f"Typ zdjęcia: {x}")
    print(f"Oryginalny rozmiar: {size}\nRozmiar po kompresji = {sizeR11}")
    print(f"Rozmiar po dekompresji = {sizeR12}")
    print("Stopień kompresji: {:.2f}".format(((size-sizeR11)/size)*100)+"%")
    print(f"Zgodność: {sizeR12 == size}\n")

    print("\tByte Run")
    print(f"Typ zdjęcia: {x}")
    print(f"Oryginalny rozmiar: {size}\nRozmiar po kompresji = {sizeB11}")
    print(f"Rozmiar po dekompresji = {sizeB12}")
    print("Stopień kompresji: {:.2f}".format(((size-sizeB11)/size)*100) + "%")
    print(f"Zgodność: {sizeB12 == size}\n\n")

