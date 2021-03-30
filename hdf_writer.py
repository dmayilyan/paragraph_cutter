import h5py
import matplotlib.pyplot as plt
import numpy as np

def extract_info(image_path):
    print(image_path.name)
    book, b, c = image_path.name.split("_")
    volume = "".join(i for i in b if i.isdigit())
    page = "".join(i for i in c if i.isdigit())

    return book, volume, page


def write_column(volume, im_path, column_lines):
    book, volume, page = extract_info(im_path)

    with h5py.File(f"{book}.h5", "w") as hdf:
        for _, segment_data in column_lines.items():
            hdf_path = f"volume_{volume}/{page}"
            if not hdf.__contains__(hdf_path):
                hdfgroup = hdf.create_group(f"volume_{volume}/{page}")

            segment_string = "_".join(str(i) for i in segment_data[0])
            hdfgroup.create_dataset(f"ps_{segment_string}", data=segment_data[1])


def read_h5():
    book = "HSH"
    with h5py.File(f"{book}.h5", "r") as hdf:
        for a, b in list(hdf["volume_3/255"].items()):
            plt.imshow(b)
            plt.savefig(f"{a}.png")


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
    read_h5()
