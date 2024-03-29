import cv2
import numpy as np
from tqdm import tqdm


def extract_mean_and_std_from_images(img_paths: list[str], img_dims=(224, 224)) -> tuple[float]:
    imgs = []
    means, stds = [], []

    for path in tqdm(img_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, img_dims)
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)  # shape = (244, 244, 3, n_images)
    imgs = imgs.astype(np.float32) / 255.

    # opencv channel order is BGR so we must reverse to get RGB
    for channel in [2, 1, 0]:
        channel_pixels = imgs[:, :, channel, :].ravel()
        means.append(channel_pixels.mean())
        stds.append(channel_pixels.std())

    return means, stds
