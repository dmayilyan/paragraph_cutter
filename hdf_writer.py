import h5py


def flatten_image(im):
    if im.shape[1] != 1:
        print("Image is not b/w")
        return

    return im.flatten()


def main(path):
    dataset = file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    meta_set = file.create_dataset("meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels)
    with h5py.File("HSH.hdf5", "w") as f:
        dset = f.create_dataset(f"{path.name}[:-4]", (100,), dtype="i")


if __name__ == "__main__":
    main()
