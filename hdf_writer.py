import h5py
import numpy as np

def flatten_image(im):
    if im.shape[1] != 1:
        print("Image is not b/w")
        return

    return im.flatten()


def write_page(page):
    print(page.items())
    ...

def write_column(col):
    ...



def main(volume=None, page=None, segment=None):
    if not volume:
        print("Volume is not selected")
    # dataset = file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    # meta_set = file.create_dataset("meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels)
    # with h5py.File("HSH.hdf5", "w") as f:
    # dset = f.create_dataset(f"{path.name}[:-4]", (100,), dtype="i")

    page = 234
    page_segment = 12
    with h5py.File("HSH.h5", "w") as hdf:
        hdfgroup = hdf.create_group(f"volume_{volume}/{page}")
        hdfgroup.create_dataset(f"ps_{page_segment}", data=np.array([1, 2, 3]))

    with h5py.File("HSH.h5", "r") as hdf:
        print(list(hdf.items()))


if __name__ == "__main__":  # pragma: no cover
    volume: int = 3
    page: int = 123
    segment: list = [[2, 2], [5, 6]]
    main(volume, page, segment)
