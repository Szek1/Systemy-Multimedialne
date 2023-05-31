import numpy as np
import matplotlib.pyplot as plt
import cv2


def imgToFloat(data):
    if (np.issubdtype(data.dtype, np.floating) == False):
        data = data.astype('float32') / 255
        return data

def colorFit(pixel, pallet):
    # print(np.linalg.norm(pallet - pixel,axis = 1))
    return pallet[np.argmin(np.linalg.norm(pallet - pixel, axis=1))]
paleta = np.linspace(0,1,3).reshape(3,1)
print(colorFit(0.43,paleta)) # 0.5
print(colorFit(0.66,paleta)) # ?
print(colorFit(0.8,paleta)) # ?

pallet8 = np.array([
    [0.0, 0.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 1.0, ],
    [1.0, 0.0, 0.0, ],
    [1.0, 0.0, 1.0, ],
    [1.0, 1.0, 0.0, ],
    [1.0, 1.0, 1.0, ],
])

pallet16 = np.array([
    [0.0, 0.0, 0.0, ],
    [0.0, 1.0, 1.0, ],
    [0.0, 0.0, 1.0, ],
    [1.0, 0.0, 1.0, ],
    [0.0, 0.5, 0.0, ],
    [0.5, 0.5, 0.5, ],
    [0.0, 1.0, 0.0, ],
    [0.5, 0.0, 0.0, ],
    [0.0, 0.0, 0.5, ],
    [0.5, 0.5, 0.0, ],
    [0.5, 0.0, 0.5, ],
    [1.0, 0.0, 0.0, ],
    [0.75, 0.75, 0.75, ],
    [0.0, 0.5, 0.5, ],
    [1.0, 1.0, 1.0, ],
    [1.0, 1.0, 0.0, ]
])


def kwant_colorFit(img, pallet):
    img = imgToFloat(img)
    height,width = img.shape[:2]
    out_img = img.copy()
    for r in range(height):
        for c in range(width):
            out_img[r, c] = colorFit(img[r, c], pallet)
    return out_img


def random_dithering(img):
    img = imgToFloat(img)
    out_img = img.copy()
    if(img.shape == 2):
        height, width = img.shape[:2]
        rand = np.random.rand(height, width)
        out_img = img >= rand
        out_img = img * 1
    else:
        height, width,channel = img.shape
        rand = np.random.rand(height, width,channel)
        for r in range(height):
            for c in range(width):
                out_img[r, c] = out_img[r, c] >= rand[r, c]
                out_img[r, c] *= 1

    return out_img


def organized_dithering(img, pallet):
    img = imgToFloat(img)
    out_img = img.copy()
    height, width = img.shape[:2]
    M = np.array([[0, 8, 2, 10],
                  [12, 4, 14, 6],
                  [3, 11, 1, 9],
                  [15, 7, 13, 5]])
    M =  (1 / 16) * M
    Mpre = (M + 1)/3 - 0.5 # Nie wiem czemu tutaj to z dzieleniem na 4 nie działa dobrze
    # jakieś czarne wychodza nie wiem o co chodzi
    for r in range(height):
        for c in range(width):
            out_img[r, c] = colorFit(out_img[r, c] + Mpre[r % 4, c % 4], pallet)
    return out_img


def floyd_steinberg_dithering(img, pallet):
    img = imgToFloat(img)
    height, width = img.shape[:2]
    out_image = img.copy()
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            oldpixel = out_image[r, c].copy()
            newpixel = colorFit(oldpixel, pallet)
            out_image[r, c] = newpixel
            quant_error = oldpixel - newpixel
            out_image[r + 1, c] = out_image[r + 1, c] + quant_error * 7 / 16
            out_image[r - 1, c + 1] = out_image[r - 1, c + 1] + quant_error * 3 / 16
            out_image[r, c + 1] = out_image[r, c + 1] + quant_error * 5 / 16
            out_image[r + 1, c + 1] = out_image[r + 1, c + 1] + quant_error * 1 / 16
    out_image = np.clip(out_image,0,1)
    return out_image


# print(colorFit(np.array([0.25,0.25,0.5]),pallet8))
# print(colorFit(np.array([0.25,0.25,0.5]),pallet16))

# img = plt.imread("0008.png")
#
# pallet_8bit = np.linspace(0, 1, 16).reshape(16, 1)
# pallet_4bit = np.linspace(0, 1, 8).reshape(8, 1)
# pallet_2bit = np.linspace(0, 1, 4).reshape(4, 1)
# pallet_1bit = np.linspace(0, 1, 2).reshape(2, 1)
#
# palety = [pallet_8bit,pallet_4bit,pallet_2bit,pallet_1bit]
# nazwy = ["8 bitów", "4 bity", "2 bity", "1 bit"]
# i = 1
# for i in range(len(palety)):
#     plt.subplot(2,2,i+1)
#     plt.title(nazwy[i])
#     plt.imshow(kwant_colorFit(img,palety[i]))
# plt.show()

### 2

pallet_4bit = np.linspace(0, 1, 8).reshape(8, 1)
pallet_2bit = np.linspace(0, 1, 4).reshape(4, 1)
pallet_1bit = np.linspace(0, 1, 2).reshape(2, 1)

# Orginal
# kwant_colorFit()
# random_dithering()
# organized_dithering()
# floyd_steinberg_dithering()


img_2bit = cv2.imread("0008.png")
plt.subplot(2,3,1)
plt.title("Oryginalny")
plt.imshow(img_2bit)

plt.subplot(2,3,2)
plt.title("Czysta kwantyzacja")
plt.imshow(kwant_colorFit(img_2bit,pallet_1bit),cmap='gray')
#
plt.subplot(2,3,3)
plt.title("Dithering losowy")
plt.imshow(random_dithering(img_2bit),cmap='gray')


plt.subplot(2,3,4)
plt.title("Dithering zorganizowany")
plt.imshow(organized_dithering(img_2bit,pallet_1bit),cmap='gray')

plt.subplot(2,3,6)
plt.title("Dithering Steinberga")
plt.imshow(floyd_steinberg_dithering(img_2bit,pallet_1bit),cmap='gray')

plt.show()

# img_2bit = cv2.imread("0007.tif")
# plt.subplot(2,3,1)
# plt.title("Oryginalny")
# plt.imshow(img_2bit)
#
# plt.subplot(2,3,2)
# plt.title("Czysta kwantyzacja")
# plt.imshow(kwant_colorFit(img_2bit,pallet_2bit),cmap='gray')
#
# plt.subplot(2,3,3)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img_2bit),cmap='gray')
#
# plt.subplot(2,3,4)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img_2bit,pallet_2bit),cmap='gray')
#
# plt.subplot(2,3,6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img_2bit,pallet_2bit),cmap='gray')
#
# plt.show()
# img_2bit = cv2.imread("0007.tif")
# plt.subplot(2,3,1)
# plt.title("Oryginalny")
# plt.imshow(img_2bit)
#
# plt.subplot(2,3,2)
# plt.title("Czysta kwantyzacja")
# plt.imshow(kwant_colorFit(img_2bit,pallet_4bit),cmap='gray')
#
# plt.subplot(2,3,3)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img_2bit),cmap='gray')
#
# plt.subplot(2,3,4)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img_2bit,pallet_4bit),cmap='gray')
#
# plt.subplot(2,3,6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img_2bit,pallet_4bit),cmap='gray')
#
# plt.show()
# 1bit


# img_1bit = cv2.imread("0008.png")
#
# plt.subplot(2,3,1)
# plt.title("Oryginalny")
# plt.imshow(img_1bit)
#
# plt.subplot(2,3,2)
# plt.title("Czysta kwantyzacja")
# plt.imshow(kwant_colorFit(img_1bit,pallet_1bit),cmap='gray')
#
# plt.subplot(2,3,3)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img_1bit),cmap='gray')
#
# plt.subplot(2,3,4)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img_1bit,pallet_1bit),cmap='gray')
#
# plt.subplot(2,3,6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img_1bit,pallet_1bit),cmap='gray')
#
# plt.show()
img_1bit = cv2.imread("0008.png")

plt.subplot(2,3,1)
plt.title("Oryginalny")
plt.imshow(img_1bit)

plt.subplot(2,3,2)
plt.title("Czysta kwantyzacja")
plt.imshow(kwant_colorFit(img_1bit,pallet_2bit),cmap='gray')

plt.subplot(2,3,3)
plt.title("Dithering losowy")
plt.imshow(random_dithering(img_1bit),cmap='gray')

plt.subplot(2,3,4)
plt.title("Dithering zorganizowany")
plt.imshow(organized_dithering(img_1bit,pallet_2bit),cmap='gray')

plt.subplot(2,3,6)
plt.title("Dithering Steinberga")
plt.imshow(floyd_steinberg_dithering(img_1bit,pallet_2bit),cmap='gray')

plt.show()
# img_1bit = cv2.imread("0008.png")
#
# plt.subplot(2,3,1)
# plt.title("Oryginalny")
# plt.imshow(img_1bit)
#
# plt.subplot(2,3,2)
# plt.title("Czysta kwantyzacja")
# plt.imshow(kwant_colorFit(img_1bit,pallet_4bit),cmap='gray')
#
# plt.subplot(2,3,3)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img_1bit),cmap='gray')
#
# plt.subplot(2,3,4)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img_1bit,pallet_4bit),cmap='gray')
#
# plt.subplot(2,3,6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img_1bit,pallet_4bit),cmap='gray')
#
# plt.show()

# img_1bit = cv2.imread("0006.tif")
# img_1bit = cv2.cvtColor(img_1bit, cv2.COLOR_BGR2RGB)
#
# plt.subplot(2,3,1)
# plt.title("Oryginalny")
# plt.imshow(img_1bit)
#
# plt.subplot(2,3,2)
# plt.title("Czysta kwantyzacja")
# plt.imshow(kwant_colorFit(img_1bit,pallet8))
#
# plt.subplot(2,3,3)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img_1bit))
#
# plt.subplot(2,3,4)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img_1bit,pallet8))
#
# plt.subplot(2,3,6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img_1bit,pallet8))
#
# plt.show()

# pal = np.array([
#     [0.847, 0.847, 0.847, ],
#     [0.753, 0.753, 0.753, ],
#     [0.094, 0.282, 0.565, ],
#     [0.659, 0.659, 0.659, ],
#     [0.282, 0.094, 0.094, ],
#     [0.376, 0.189, 0.189, ],
#     [0.470, 0.470, 0.565, ],
#     [0.659, 0.565, 0.470, ],
#     [0.282, 0.376, 0.565, ],
#     [0.470, 0.376, 0.282, ],
#     [0.189, 0, 0, ],
#     [0.565, 0.565, 0.376, ],
#     [0.565, 0.565, 0.659, ],
#     [0.941, 0.941, 0.847, ],
#     [1, 0.941, 0.847, ]
# ])
#
# # 15 kolorow
# img = plt.imread('0013.jpg')
#
# plt.subplot(2, 3, 1)
# plt.title("Orginalny")
# plt.imshow(img)
#
# plt.subplot(2, 3, 3)
# plt.title("Zwykla kwantyzacja")
# plt.imshow(kwant_colorFit(img, pal))
#
# plt.subplot(2, 3, 4)
# plt.title("Dithering losowy")
# plt.imshow(random_dithering(img))
#
# plt.subplot(2, 3, 5)
# plt.title("Dithering zorganizowany")
# plt.imshow(organized_dithering(img, pal))
#
# plt.subplot(2, 3, 6)
# plt.title("Dithering Steinberga")
# plt.imshow(floyd_steinberg_dithering(img, pal))
# plt.show()
