import os
import re
from math import atan2, degrees
from typing import Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from beartype import beartype
from scipy import fftpack
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from hdf_writer import write_column

volume: int = 3


@beartype
def read_config() -> dict:
    with open("HSH_config.yaml") as stream:
        return yaml.safe_load(stream)


@beartype
def plot_projection(img: Any, axis: int) -> None:
    meanh = np.mean(img, axis=axis)
    plt.plot(meanh)
    plt.xlim(100, 1750)
    plt.show()


@beartype
def filter_pages(file_list: Any, include_pages: list = None) -> list:
    pages: dict = {}
    for p in file_list:
        pages[int(re.search(r"page([\d]{1,3})", p.path).group(1))] = p

    page_numbers = sorted(pages.keys())
    if not include_pages:
        include_pages = page_numbers[5:-4]

    # black_ratio = []
    # for i, p in enumerate(file_list):
    # img = read_image(p.path)
    # ratio = np.sum(img.flatten() < 128) / len(img.flatten())
    # black_ratio.append(ratio)
    # if ratio < 0.058:
    # print(p.path, ratio)

    # plt.hist(black_ratio, bins = 50)
    # plt.savefig("black_ratio.png")

    return [v for k, v in pages.items() if k in include_pages]


# @beartype
def read_image(img_path):
    img = cv2.imread(img_path.path, cv2.IMREAD_GRAYSCALE)
    # plot_projection(img, 0)
    #     plt.figure(figsize=(img.shape[0] / 50, img.shape[1] / 50))
    #     plt.imshow(img, cmap="gray", interpolation="bicubic")
    #     plt.show()
    return img


def get_paths():
    base: str = "HSH"
    file_list = os.scandir(base)
    file_list = filter_pages(file_list)

    return file_list


@beartype
def estimate_cuts(img_paths, sample_pages: list, peak_config: dict) -> Tuple[int, int]:
    sample_pages = filter_pages(img_paths, sample_pages)

    peak_locations = []
    for p in sample_pages:
        im = read_image(p)
        im = straighten_image(im)
        # im = preprocess_image(im)
        peak_locations.append(get_peaks(im, peak_config))

    peak_locations: np.ndarray[np.int32] = np.array(peak_locations)

    return int(peak_locations[:, 0].mean()), int(peak_locations[:, 1].mean())


@beartype
def get_peaks(img, peak_config):
    # Horizontal averaging
    meanh = np.average(img, axis=0)
    smoothed = uniform_filter1d(meanh, size=8)
    # A margin is taken to remove page edge peaks
    peaks, _ = find_peaks(
        smoothed[peak_config["margin_left"] : -peak_config["margin_right"]],
        height=peak_config["height"],
        distance=peak_config["distance"],
    )

    print(f"peaks: {peaks[0] + peak_config['margin_left'], peaks[1] + peak_config['margin_left']}")
    return peaks[0] + peak_config["margin_left"], peaks[1] + peak_config["margin_left"]


@beartype
def get_columns(img, peaks):
    return img[:, : peaks[0]], img[:, peaks[0] : peaks[1]], img[:, peaks[1] :]


def crop_top_bottom(ims: list) -> list:
    tolerance: int = 240
    margin: int = 5
    top_cut = bottom_cut = 0
    cr_ims: list = []
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


@beartype
def get_horizontal_cut(mm, tolerance: int, right: bool = False) -> int:
    if right:
        mm = mm[::-1]

        plt.plot(mm)
        plt.savefig("average_plot.png")

    cut: int = -1
    for i, val in enumerate(mm):
        if val < tolerance:
            break
        if right:
            cut = mm.size - i
        else:
            cut = i

    return cut


@beartype
def crop_left_right(ims: list) -> list:
    tolerance: int = 245
    margin: int = 7
    cr_ims: list = []
    left_cut = right_cut = 0
    for im in ims:
        # We don't care about blur in other axis
        blur_img = cv2.GaussianBlur(im, (9, 1), 0)
        mm = np.mean(blur_img, axis=0)

        left_cut = get_horizontal_cut(mm, tolerance, right=False)
        right_cut = get_horizontal_cut(mm, tolerance, right=True)
        print("cuts: ", left_cut, right_cut)

        cr_ims.append(im[:, left_cut - margin : right_cut + margin])

    return cr_ims


@beartype
def is_valid_segment(im_segment) -> bool:
    count: int = len(im_segment.flatten())
    if count < 600:
        return False

    side_ratio: float = im_segment.shape[1] / im_segment.shape[0]
    if side_ratio < 5 or side_ratio > 25:
        return False

    color_ratio: float = sum(im_segment.flatten() < 128) / count
    if color_ratio < 0.11:
        return False

    return True


def crop_lines(im) -> dict:
    blur = cv2.GaussianBlur(im, (9, 1), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cut_ims: dict = {}
    counter: int = 0
    for contour in contours:
        minAreaRect = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # Margin adjustment
        y -= 2
        h += 4
        valid_cut = is_valid_segment(im[y : y + h, x : x + w])

        if valid_cut:
            cut_ims[counter] = ((x, y, w, h), im[y : y + h, x : x + w])

            # cut_ims.append(im[y : y + h, x : x + w])
            counter += 1

    # box_contours = []
    # for cont in contours:
    # min_area = cv2.minAreaRect(cont)
    # boxPoints = cv2.boxPoints(min_area)
    # box = np.int0(boxPoints)
    # box_contours.append(box)

    # fig = plt.figure(figsize=(im.shape[0] / 40, im.shape[1] / 40))
    # cont_img = cv2.drawContours(im, box_contours, -1, (0, 255, 7), 1)
    # plt.imshow(cont_img, cmap="gray", interpolation="bicubic")
    # plt.savefig("cont_page.png")

    # print(f"cont count {cont_count}")

    return cut_ims


def straighten_image(img):

    image_orig = img.copy()
    dim = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img = cv2.GaussianBlur(img, (13, 13), 0)

    edges = canny(img, 2)
    tested_angles = np.linspace(-np.pi / 15, np.pi / 15, 200, endpoint=False)
    lines = probabilistic_hough_line(
        edges, threshold=240, line_length=dim[1] - int(dim[1] * 0.2), line_gap=int(dim[0] / 3), theta=tested_angles
    )

    angles = []
    for line in lines:
        p0, p1 = line
        angle = 90 + degrees(atan2(p1[1] - p0[1], p1[0] - p0[0]))
        angles.append(angle)
        # if round(angle, 1) <= 360.0:
        # angles.append(angle)

    mean_angle = np.mean(angles)
    # print(f"mean angle: {mean_angle}")

    (h, w) = image_orig.shape[:2]

    center = (w // 2, h // 2)

    # perform the rotation
    mat = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
    return cv2.warpAffine(image_orig, mat, (w, h), borderValue=255)


def preprocess_image(img):
    tolerance: int = 220
    margin: int = 9
    blur_img = cv2.GaussianBlur(img, (9, 1), 0)
    mm = np.mean(blur_img, axis=0)
    left_cut = get_horizontal_cut(mm, tolerance, right=False)

    return img[:, left_cut - margin :]


def process_images(img, config, im_path, cuts) -> None:
    peak_config = config[volume]["peaks"]

    img = straighten_image(img)
    peaks = get_peaks(img, peak_config)
    ims: list = get_columns(img, peaks)
    # fig = plt.figure()
    # plt.imshow(ims[0])
    # plt.savefig("qwe1.png")
    ims: list = crop_top_bottom(ims)
    ims: list = crop_left_right(ims)

    # fig = plt.figure(figsize=(img.shape[0] / 50, img.shape[1] / 50))
    # plt.imshow(img, cmap="gray")
    # plt.savefig("page.png")

    #  for i, col in enumerate(ims):
    #  fig = plt.figure(figsize=(col.shape[1] / 50, col.shape[0] / 50))
    #  plt.imshow(col, cmap="gray")
    #  plt.tight_layout()
    #  plt.savefig(f"col_{i}.png")

    cropped_lines: list = []
    max_workers: int = 4
    # with Pool(max_workers) as p:
    # cropped_lines.append(p.map(crop_lines, ims))
    for im in ims[1:2]:
        column_lines = crop_lines(im)
        cropped_lines.append(column_lines)
        write_column(volume, im_path, column_lines)  # TODO

    # fig = plt.figure(figsize=(40, 20))
    # for i, line in enumerate(cropped_lines[0]):
    # plt.subplot(20, 11, i + 1)
    # plt.imshow(cropped_lines[0][i], cmap="gray")
    # plt.title(
    # f"{i} count: {len(line.flatten())}, s_ratio: {line.shape[1] / line.shape[0]:.0f} "
    # f"c_ratio: {sum(line.flatten() < 128) / len(line.flatten()):.2f}"
    # )
    # plt.tight_layout()
    # plt.savefig("qwe.png")

    # fig = plt.figure(figsize=(img.shape[0] / 100, img.shape[1] / 100))
    # fig.suptitle("Cuts of the page")
    # gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0, hspace=0)
    # axs = gs.subplots(sharey="row")

    # for ax, im in zip(axs, ims):
    # ax.imshow(im, cmap="gray", interpolation="bicubic")
    # ax.label_outer()

    # plt.show()


def get_zscore(im: Any, freq_cut, window_size):
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


def main() -> None:
    img_paths = get_paths()
    config: dict = read_config()
    cuts = estimate_cuts(img_paths, config[volume]["sample_pages"], config[volume]["peaks"])
    for im_path in img_paths[18:19]:
        im = read_image(im_path)
        process_images(im, config, im_path, cuts)


if __name__ == "__main__":  # pragma: no cover
    main()
