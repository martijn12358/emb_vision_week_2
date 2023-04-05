import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

img = plt.imread("mri_snapshot.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def ass_1():
    hist = ndi.histogram(img, 0, 255, 256)
    hist_n = hist / hist.sum()
    img_equ = cv2.equalizeHist(img)
    hist_equ = ndi.histogram(img_equ, 0, 255, 256)
    hist_equ_n = hist_equ / hist_equ.sum()
    cdf = hist_n.cumsum()
    cdf_n = cdf * float(hist_n.max()) / cdf.max()
    cdf_equ = hist_equ_n.cumsum()
    cdf_n_equ = cdf_equ * float(hist_equ_n.max()) / cdf_equ.max()

    fig, axes = plt.subplots(2, 2)
    axes[1][1].plot(hist_equ_n, label='Histogram')
    axes[1][0].plot(hist_n, label='Histogram')
    axes[1][1].plot(cdf_n_equ)
    axes[1][0].plot(cdf_n)
    axes[0][1].imshow(img_equ, cmap='gray' )
    axes[0][0].imshow(img, cmap='gray')
    axes[0][0].axis('off')
    axes[0][1].axis('off')
    plt.show()

def ass_2():
    kernel = np.ones((3, 3)) / 9
    filtered_img = ndi.convolve(img, kernel)
    kernel2 = np.ones((7, 7)) / 49
    filtered_img2 = ndi.convolve(img, kernel2)

    fig, axes = plt.subplots(1, 3)
    axes[1].imshow(filtered_img, cmap='gray')
    axes[0].imshow(img, cmap='gray')
    axes[2].imshow(filtered_img2, cmap='gray')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    plt.show()


def ass_3():
    filtered_img = ndi.median_filter(img, size=3)
    filtered_img2 = ndi.median_filter(img, size=7)

    fig, axes = plt.subplots(1, 3)
    axes[1].imshow(filtered_img, cmap='gray')
    axes[0].imshow(img, cmap='gray')
    axes[2].imshow(filtered_img2, cmap='gray')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    plt.show()


def ass_4():
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(img, -1, kernel)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[1].imshow(sharp, cmap='gray')
    axes[0].axis('off')
    axes[1].axis('off')
    plt.show()

ass_4()

