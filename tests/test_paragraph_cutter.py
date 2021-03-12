import os

from pytest import fixture
from pytest_mock import MockerFixture, mocker

from paragraph_cutter import get_paths, filter_pages


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


def test_get_paths(mocker: MockerFixture):
    mocker.patch("os.scandir", return_value=FileList, autospec=True)
    assert get_paths() == ["qwe", "123"]


def test_filter_pages_include():
    file_list = FileList()
    assert len(filter_pages(file_list, include_pages=[20, 101])) == 2
    assert list(filter_pages(file_list, include_pages=[20]))[0].path == "HSH/test_page20.jpg"
    assert list(filter_pages(file_list, include_pages=[20]))[0].name == "test_page20.jpg"


def test_filter_pages_None():
    file_list = FileList()
    assert len(filter_pages(file_list)) == 114


if __name__ == "__main__":
    filelist = FileList()
    print(type(i) for i in FileList())
    for i in filelist:
        print(i.name, i.path)
