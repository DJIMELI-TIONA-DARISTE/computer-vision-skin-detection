import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from math import *
import os
from mpl_toolkits import mplot3d
ECHELLE = 128
SEUIL = 100

# upload images and masks

path_original_img_train = 'DATASET/Dataset8_Abdomen/train/original_images'
path_masks_img_train = 'DATASET/Dataset8_Abdomen/train/skin_masks'

liste_images = [ f for f in os.listdir(path_original_img_train) if os.path.isfile(os.path.join(path_original_img_train,f)) ]

t_liste_images_originals = []
t_liste_images_masks = []

for img in liste_images:
    image = cv.imread(os.path.join(path_original_img_train, img))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    t_liste_images_originals.append(image)
    
for img in liste_images:
    image_mask = cv.imread(os.path.join(path_masks_img_train, img.split('.')[0]+'.png'), cv.IMREAD_GRAYSCALE)
    t_liste_images_masks.append(image_mask)
    
t_liste_images_originals = np.array(t_liste_images_originals)
t_liste_images_masks = np.array(t_liste_images_masks)
print(t_liste_images_originals.shape)
print(t_liste_images_masks.shape)

# On affiche quelques images et leurs masques

plt.figure(figsize=(10, 10))
for k, i in np.ndenumerate(np.random.randint(t_liste_images_originals.shape[0], size=6)):
    ax = plt.subplot(4, 4, 2 * k[0] + 1)
    plt.imshow(t_liste_images_originals[i], cmap='gray')
    plt.title('image {}'.format(i))
    plt.axis("off")
    x = plt.subplot(4, 4, 2 * k[0] + 2)
    plt.imshow(t_liste_images_masks[i], cmap='gray')
    plt.title('masque {}'.format(i))
    plt.axis("off")
# On change l'espace de couleur de RGB à Lab

def RGB_to_Lab(t_RGB):
    t_lab = [0] * t_RGB.shape[0]
    for i, img in enumerate(t_RGB):
        t_lab[i] = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    return np.array(t_lab)
t_liste_images_lab = RGB_to_Lab(t_liste_images_originals)
# Première image dans l'espace Lab

i = 13

plt.figure(figsize=(10, 10))
ax = plt.subplot(4, 4, 1)
plt.imshow(t_liste_images_originals[i])
plt.title('image RGB {}'.format(i))
plt.axis("off")

ax = plt.subplot(4, 4, 2)
plt.imshow(t_liste_images_lab[i])
plt.title('image Lab {}'.format(i))
plt.axis("off")

ax = plt.subplot(4, 4, 3)
plt.imshow(t_liste_images_masks[i], cmap='gray')
plt.title('masque {}'.format(i))
plt.axis("off")
l_channel,a_channel,b_channel = cv.split(t_liste_images_lab[13])

print()
print('valeur minimale d\'un dans le chanel L : {}'.format(np.array(l_channel).min()))
print('valeur maximale d\'un dans le chanel L : {}'.format(np.array(l_channel).max()))
print()
print('valeur minimale d\'un dans le chanel a : {}'.format(np.array(a_channel).min()))
print('valeur maximale d\'un dans le chanel a : {}'.format(np.array(a_channel).max()))
print()
print('valeur minimale d\'un dans le chanel b : {}'.format(np.array(b_channel).min()))
print('valeur maximale d\'un dans le chanel b : {}'.format(np.array(b_channel).max()))
# On convertie l'intervalle dans lequel les pixels prennent leurs valeurs pour les dimensions a et b

def convertLab8toLabx(t_img):
    temp = [0] * 256
    for i in range(256):
        temp[i] = floor(i / (256/ECHELLE))
    
    for index, img in enumerate(t_img):
        h, w, d = img.shape
        image = np.asarray(np.zeros((h, w, d), dtype=np.uint8))

        for i in range(h):
            for j in range(w):
                for k in range(1, 3):
                    image[i, j][k] = temp[img[i, j][k]]
                image[i, j][0] = img[i, j][0]
        t_img[index] = image
        
    return t_img
img_lab_convert = convertLab8toLabx(t_liste_images_lab)

l_channel,a_channel,b_channel = cv.split(img_lab_convert[13])

print()
print('valeur minimale d\'un dans le chanel L : {}'.format(np.array(l_channel).min()))
print('valeur maximale d\'un dans le chanel L : {}'.format(np.array(l_channel).max()))
print()
print('valeur minimale d\'un dans le chanel a : {}'.format(np.array(a_channel).min()))
print('valeur maximale d\'un dans le chanel a : {}'.format(np.array(a_channel).max()))
print()
print('valeur minimale d\'un dans le chanel b : {}'.format(np.array(b_channel).min()))
print('valeur maximale d\'un dans le chanel b : {}'.format(np.array(b_channel).max()))
# Première image dans l'espace Lab avec les intervalles de a et b convertis

i = 0

plt.figure(figsize=(10, 10))
ax = plt.subplot(4, 4, 1)
plt.imshow(img_lab_convert[i])
plt.title('image {}'.format(i))
plt.axis("off")

ax = plt.subplot(4, 4, 2)
plt.imshow(t_liste_images_masks[i], cmap='gray')
plt.title('image {}'.format(i))
plt.axis("off")

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()