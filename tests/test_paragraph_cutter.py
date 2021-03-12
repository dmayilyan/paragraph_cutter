import os

from pytest import fixture
from pytest_mock import MockerFixture, mocker

from paragraph_cutter import get_paths


class FileList:
    def __init__(self):
        self.name = ["test_page1.jpg", "test_page20.jpg", "test_page231.jpg"]
        self.path = ["HSH/test_page1.jpg", "HSH/test_page20.jpg", "HSH/test_page231.jpg"]

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


if __name__ == "__main__":
    filelist = FileList()
    for i in filelist:
        print(i.name, i.path)
