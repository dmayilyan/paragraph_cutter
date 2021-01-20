import os

import cv2
import numpy as np
from scipy import fftpack


def read_image():
    print(os.listdir("./HSH"))
    img = cv2.imread("./HSH/HSH_vol1_page14.jpg", cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(img.shape[0] / 50, img.shape[1] / 50))
    plt.imshow(img, cmap="gray", interpolation="bicubic")


def cut_page(img):
    CUT1 = 354
    CUT2 = 677

    return img[:, 0:CUT1], img[:, CUT1:CUT2], img[:, CUT2:]


def crop_top_bottom(ims):
    TOL = 220
    MARGIN = 5
    cr_ims = []
    for im in ims:
        blur_img = cv2.GaussianBlur(im, (27, 27), 0)
        mm = np.mean(blur_img, axis=1)
        for i, val in enumerate(mm):
            if val < TOL:
                break
            else:
                top_cut = i

        for i, val in enumerate(mm[::-1]):
            if val < TOL:
                break
            bottom_cut = mm.size - i

        cr_ims.append(im[top_cut - MARGIN : bottom_cut + MARGIN, :])

    return cr_ims


def crop_left_right(ims):
    TOL = 220
    MARGIN = 5
    cr_ims = []
    for im in ims:
        blur_img = cv2.GaussianBlur(im, (9, 9), 0)
        mm = np.mean(blur_img, axis=0)
        for i, val in enumerate(mm):
            if val < TOL:
                break
            left_cut = i

        for i, val in enumerate(mm[::-1]):
            if val < TOL:
                break
            else:
                right_cut = mm.size - i

        cr_ims.append(im[:, left_cut - MARGIN : right_cut + MARGIN])

    return cr_ims


def process_images():
    ims = cut_page(img)
    ims = crop_top_bottom(ims)
    ims = crop_left_right(ims)

    fig = plt.figure(figsize=(img.shape[0] / 100, img.shape[1] / 100))
    fig.suptitle("Cuts of the page")
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0, hspace=0)
    axs = gs.subplots(sharey="row")

    for ax, im in zip(axs, ims):
        ax.imshow(im, cmap="gray", interpolation="bicubic")
        ax.label_outer()


def get_zscore(im, freq_cut, window_size):
    meanv = np.mean(im, axis=1)
    transformed = fftpack.fft(meanv)

    #     smoothed = uniform_filter1d(meanv, size=1)
    signal = fftpack.fft(meanv)
    W = fftpack.fftfreq(signal.size, d=1 / signal.size)
    cut_f_signal = signal.copy()

    cut_f_signal[(np.abs(W) < freq_cut)] = 0

    cut_signal = fftpack.ifft(cut_f_signal)

    signal_pd = pd.DataFrame(signal_series, columns=["data"])
    signal_pd["std_col"] = signal_pd["data"].rolling(window=window_size, center=True).std()
    signal_pd["mean_col"] = signal_pd["data"].rolling(window=window_size, center=True).mean()
    signal_pd["zscore"] = (signal_pd["data"] - signal_pd["mean_col"]) / signal_pd["std_col"]

    return signal_pd["zscore"]


if __name__ == "__main__":
    read_image()
