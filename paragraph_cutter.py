import os

import cv2
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def plot_projection(img, axis):
    meanh = np.average(img, axis=axis)
    plt.plot(meanh)
    plt.xlim(100, 1750)
    plt.show()


def read_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plot_projection(img, 0)
#     plt.figure(figsize=(img.shape[0] / 50, img.shape[1] / 50))
#     plt.imshow(img, cmap="gray", interpolation="bicubic")
#     plt.show()
    return img


def read_images():
    base = "HSH"
    file_list = os.listdir(base)
    return read_image(f"{base}/{file_list[1]}")


def cut_page(img):
    # Horizontal averaging
    meanh = np.average(img, axis=0)
    smoothed = uniform_filter1d(meanh, size=10)
    # A margin is taken to remove page edge peaks
    peaks, _ = find_peaks(smoothed[100:-100], height=230, distance=400)

    return img[:, 0:peaks[0]], img[:, peaks[0]:peaks[1]], img[:, peaks[1]:]


def crop_top_bottom(ims):
    TOL = 220
    MARGIN = 5
    top_cut = bottom_cut = 0
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

        cr_ims.append(im[top_cut - MARGIN: bottom_cut + MARGIN, :])

    return cr_ims


def crop_left_right(ims):
    TOL = 220
    MARGIN = 5
    cr_ims = []
    left_cut = right_cut = 0
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
            right_cut = mm.size - i

        cr_ims.append(im[:, left_cut - MARGIN: right_cut + MARGIN])

    return cr_ims


def process_images(img):
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

    plt.show()


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
    img = read_images()
    ims = cut_page(img)
    process_images(img)
