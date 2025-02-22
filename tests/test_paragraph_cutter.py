import cv2
import numpy as np
from pyannotate_runtime import collect_types
from pytest import fixture, mark

from paragraph_cutter import estimate_cuts, filter_pages, get_columns

collect_types.init_types_collection()

collect_types.start()


@fixture
def sample_config():
    return {
        3: {
            "sample_pages": [9, 11, 12, 14, 15, 16, 19, 33, 38, 40, 41],
            "peaks": {"margin_left": 200, "margin_right": 200, "height": 240, "distance": 400},
        }
    }


class FileList:
    def __init__(self):
        self.name = [f"test_page{i}.jpg" for i in range(123)]
        self.path = [f"HSH/test_page{i}.jpg" for i in range(123)]

    def __iter__(self):
        return FileListIterator(self)


class FileListIterator:
    def __init__(self, plist):
        self._index = 0
        self._plist = plist

    def __next__(self):
        if self._index < len(self._plist.name):
            result = DirEntry(list(zip(self._plist.name, self._plist.path))[self._index])
            self._index += 1

            return result

        raise StopIteration


class DirEntry:
    def __init__(self, path_tuple):
        self.name = path_tuple[0]
        self.path = path_tuple[1]

    def __call__(self):
        return self


@fixture
def file_list():
    return FileList()


@fixture
def image():
    np_array = np.ones((2, 333))
    cv2.imwrite("tmp_image.png", np_array)
    return cv2.imread("tmp_image.png")


# def test_get_paths(mocker: MockerFixture):
# mocker.patch("os.scandir", return_value=FileList, autospec=True)
# assert get_paths() == ["qwe", "123"]


def test_filter_pages_include(file_list):
    assert len(filter_pages(file_list, include_pages=[20, 101])) == 2
    assert list(filter_pages(file_list, include_pages=[20]))[0].path == "HSH/test_page20.jpg"
    assert list(filter_pages(file_list, include_pages=[20]))[0].name == "test_page20.jpg"


def test_filter_pages_none(file_list):
    assert len(filter_pages(file_list)) == 114


def test_get_columns(image):
    assert len(get_columns(image, [11, 22])) == 3


@mark.skip(reason="Need prior setup to test this.")
def test_estimate_cuts(sample_config, file_list):
    volume = 3
    peak_pages = sample_config[volume]["peaks"]
    sample_pages = sample_config[volume]["sample_pages"]

    cuts = estimate_cuts(file_list, sample_pages, peak_pages)  # TODO
    print(cuts)
    #  assert cuts == 3


collect_types.stop()
collect_types.dump_stats("types.json")


if __name__ == "__main__":
    image()
