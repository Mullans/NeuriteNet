import cv2
import numpy as np


def overlay_map(image, gradmap, rescale_max=None, split_signs=False):
    image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)
    image = np.dstack([image, image, image])

    gradmap = np.squeeze(gradmap)
    if image.shape != gradmap.shape:
        gradmap = cv2.resize(gradmap, tuple(image.shape[:2][::-1]))
    if rescale_max is None:
        rescale_max = np.max(np.abs(gradmap))

    if split_signs:
        negative = np.abs(np.clip(gradmap, -np.inf, 0))
        positive = np.clip(gradmap, 0, np.inf)
        gradmap = np.dstack([negative, positive, np.zeros_like(negative)])
        gradmap /= rescale_max
        gradmap = (gradmap * 255).astype(np.uint8)
    else:
        gradmap /= rescale_max
        gradmap = (gradmap * 255).astype(np.uint8)
        # gradmap = np.dstack([np.zeros_like(gradmap), gradmap, np.zeros_like(gradmap)])
        gradmap = cv2.applyColorMap(gradmap, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]

    overlay = cv2.addWeighted(image, 0.8, gradmap, 0.8, 0)
    overlay = np.where(gradmap.sum(axis=2, keepdims=True) == 0, image, overlay)
    return overlay
