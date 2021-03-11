import os
import re
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import fftpack
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def read_config():
    with open("HSH_config.yaml") as stream:
        return yaml.safe_load(stream)


def plot_projection(img, axis):
    meanh = np.average(img, axis=axis)
    plt.plot(meanh)
    plt.xlim(100, 1750)
    plt.show()


def filter_pages(file_list, exclude_pages=None):
    pages = {}
    for p in file_list:
        pages[int(re.search(r"page([\d]{1,3})", p.path).group(1))] = p

    page_numbers = sorted(pages.keys())
    if not exclude_pages:
        exclude_pages = page_numbers[5:] + page_numbers[:-4]

    # black_ratio = []
    # for i, p in enumerate(file_list):
    # img = read_image(p.path)
    # ratio = np.sum(img.flatten() < 128) / len(img.flatten())
    # black_ratio.append(ratio)
    # if ratio < 0.058:
    # print(p.path, ratio)

    # plt.hist(black_ratio, bins = 50)
    # plt.savefig("black_ratio.png")

    return [v for k, v in pages.items() if k in exclude_pages]


def read_image(img_path: str):
    img = cv2.imread(img_path.path, cv2.IMREAD_GRAYSCALE)
    # plot_projection(img, 0)
    #     plt.figure(figsize=(img.shape[0] / 50, img.shape[1] / 50))
    #     plt.imshow(img, cmap="gray", interpolation="bicubic")
    #     plt.show()
    return img


def get_paths():
    base = "HSH"
    file_list = os.scandir(base)
    file_list = filter_pages(file_list)

    return file_list


def estimate_cuts(img_paths, sample_pages: list, peak_config: dict):
    sample_pages = filter_pages(img_paths, sample_pages)
    peak_locations = []
    for p in sample_pages:
        im = read_image(p)
        peak_locations.append(get_peaks(im, peak_config))

    peak_locations = np.array(peak_locations)

    return int(peak_locations[:, 0].mean()), int(peak_locations[:, 1].mean())


def get_peaks(img, peak_config):
    print(img.shape)
    # Horizontal averaging
    meanh = np.average(img, axis=0)
    smoothed = uniform_filter1d(meanh, size=10)
    # A margin is taken to remove page edge peaks
    peaks, _ = find_peaks(
        smoothed[peak_config["margin_left"] : -peak_config["margin_right"]],
        height=peak_config["height"],
        distance=peak_config["distance"],
    )

    print(f"peaks: {peaks[0] + peak_config['margin_left'], peaks[1] + peak_config['margin_left']}")
    return peaks[0] + peak_config["margin_left"], peaks[1] + peak_config["margin_left"] 


def get_columns(img, peaks):
    return img[:, 0 : peaks[0]], img[:, peaks[0] : peaks[1]], img[:, peaks[1] :]


def crop_top_bottom(ims):
    tolerance = 220
    margin = 5
    top_cut = bottom_cut = 0
    cr_ims = []
    for im in ims:
        # We don't care about blur in other axis
        blur_img = cv2.GaussianBlur(im, (1, 27), 0)
        mm = np.mean(blur_img, axis=1)
        for i, val in enumerate(mm):
            if val < tolerance:
                break
            else:
                top_cut = i

        for i, val in enumerate(mm[::-1]):
            if val < tolerance:
                break
            bottom_cut = mm.size - i

        cr_ims.append(im[top_cut - margin : bottom_cut + margin, :])

    return cr_ims


def crop_left_right(ims):
    tolerance = 220
    margin = 5
    cr_ims = []
    left_cut = right_cut = 0
    for im in ims:
        # We don't care about blur in other axis
        blur_img = cv2.GaussianBlur(im, (9, 1), 0)
        mm = np.mean(blur_img, axis=0)
        for i, val in enumerate(mm):
            if val < tolerance:
                break
            left_cut = i

        for i, val in enumerate(mm[::-1]):
            if val < tolerance:
                break
            right_cut = mm.size - i

        cr_ims.append(im[:, left_cut - margin : right_cut + margin])

    return cr_ims


def is_valid_segment(im_segment):
    count = len(im_segment.flatten())
    if count < 600:
        return False

    side_ratio = im_segment.shape[1] / im_segment.shape[0]
    if side_ratio < 5:
        return False

    color_ratio = sum(im_segment.flatten() < 128) / count
    if color_ratio < 0.16:
        return False

    return True


def crop_lines(im):
    blur = cv2.GaussianBlur(im, (9, 1), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cut_ims = []
    for contour in contours:
        minAreaRect = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # Margin adjustment
        y -= 2
        h += 4
        val_cut = is_valid_segment(im[y : y + h, x : x + w])
        # if val_cut:
        cut_ims.append(im[y : y + h, x : x + w])

    return cut_ims


def process_images(img, config, cuts):
    peaks = get_peaks(img, config)
    ims = get_columns(img, peaks)
    fig = plt.figure()
    plt.imshow(ims[0])
    plt.savefig("qwe1.png")
    ims = crop_top_bottom(ims)
    ims = crop_left_right(ims)

    # fig = plt.figure()
    # plt.imshow(ims[0])
    # plt.savefig("qwe.png")

    cropped_lines = []
    max_workers = 4
    # with Pool(max_workers) as p:
    # cropped_lines.append(p.map(crop_lines, ims))
    for im in ims:
        cropped_lines.append(crop_lines(im))

    fig = plt.figure()
    plt.imshow(cropped_lines[0][3], cmap="gray")
    plt.savefig("qwe.png")

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
    volume = 3
    img_paths = get_paths()
    config = read_config()
    cuts = estimate_cuts(img_paths, config[volume]["sample_pages"], config[volume]["peaks"])
    for im_path in img_paths[17:18]:
        im = read_image(im_path)
        print(im_path.name)
        process_images(im, config[volume]["peaks"], cuts)
