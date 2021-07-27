import cv2
import numpy as np
import skimage
import tqdm.auto as tqdm

from .utils import enforce_4D


def _get_occlusions(image, num_patches=25, percent_overlap=0.5, background_value=0, **kwargs):
    image = enforce_4D(image)
    if percent_overlap >= 1 or percent_overlap <= 0:
        raise ValueError("Overlap must be a fraction between 0 and 1")
    if not hasattr(num_patches, '__len__'):
        num_patches = [num_patches, num_patches]
    num_patches = np.array(num_patches).astype(int)
    image_shape = image.shape[1:3]
    patch_size = image_shape / (num_patches - percent_overlap * num_patches + percent_overlap)

    locations = []
    images = []
    row_step, col_step = patch_size * (1 - percent_overlap)
    for row in range(num_patches[0]):
        if row == num_patches[0] - 1:
            row_slice = slice(int(row * row_step), int(image_shape[0]))
        else:
            row_slice = slice(int(row * row_step), int(row * row_step + patch_size[0]))
        for col in range(num_patches[1]):
            if col == num_patches[1] - 1:
                col_slice = slice(int(col * col_step), int(image_shape[1]))
            else:
                col_slice = slice(int(col * col_step), int(col * col_step + patch_size[1]))
            locations.append([row_slice, col_slice])
            new_img = np.copy(image)
            new_img[:, row_slice, col_slice] = background_value
            images.append(new_img)
    images = np.concatenate(images, axis=0)
    return images, locations


def get_occlusion_map(input_value, model, expected_label=1, class_of_interest=0, batch_size=32, **kwargs):
    model_input = enforce_4D(input_value)
    model_inputs, locations = _get_occlusions(model_input, **kwargs)
    baseline = model(model_input)[:, class_of_interest]
    batches = [model_inputs[i:i + batch_size] for i in range(0, model_inputs.shape[0], batch_size)]
    results = np.concatenate([model(batch)[:, class_of_interest] for batch in batches], axis=0)
    cam = np.zeros(model_input.shape[1:3])
    counts = np.zeros(model_input.shape[1:3])
    for idx, loc in enumerate(locations):
        cam[loc[0], loc[1]] += baseline - results[idx]
        counts[loc[0], loc[1]] += 1
    cam = np.divide(cam, counts)
    return cam


def _get_felzenswalb(input_value, scale_values=[150, 250, 500, 1000], sigma_values=[0.7], min_segment_size=400, dilation_radius=3, **kwargs):
    input_value = np.squeeze(input_value)
    selem = skimage.morphology.disk(dilation_radius)
    all_masks = []
    for scale in scale_values:
        for sigma in sigma_values:
            segments = skimage.segmentation.felzenszwalb(input_value, scale=scale, sigma=sigma, min_size=min_segment_size)
            masks = [segments == val for val in range(1, int(segments.max()))]
            masks = [cv2.dilate(item.astype(np.uint8), selem).astype(np.bool) for item in masks]
            all_masks.extend(masks)
    return all_masks


def _get_watershed(input_value, markers=100, compactness=0.0001, dilation_radius=3, **kwargs):
    input_value = np.squeeze(input_value)
    selem = skimage.morphology.disk(dilation_radius)
    segments = skimage.segmentation.watershed(input_value, markers=markers, compactness=compactness)
    masks = [segments == val for val in range(int(segments.min()), int(segments.max()))]
    masks = [cv2.dilate(item.astype(np.uint8), selem).astype(np.bool) for item in masks]
    return masks


def xrai(input_value, model, min_segment_size=400, coverage=1.0, max_segment_percent=0.10, return_raw_map=True, verbose=True, **kwargs):
    """Get the XRAI attribution for an image and model

    Parameters
    ----------
    input_value : numpy.ndarray | tensorflow.Tensor
        The image to find the attribution for
    model : tf.keras.Model
        The model to find the attribution for
    min_segment_size : int
        The minimum pixel size of segments for the felzenszwalb segmentation
    coverage : float
        The minimum percent of the image to cover with XRAI segments
    max_segment_percent : float
        The maximum percent of the image that a single segment can cover
    return_raw_map : bool
        Whether to return the raw occlusion attribution
    verbose : bool
        Whether to show a progress bar for the attribution
    """
    input_value = enforce_4D(input_value)
    attr_map = get_occlusion_map(input_value, model, **kwargs)
    segments = _get_felzenswalb(input_value, min_segment_size=min_segment_size, **kwargs)
    output_attr = np.ones(attr_map.shape, dtype=np.float) * -np.inf
    n_masks = len(segments)
    if verbose:
        print("{} segments found".format(n_masks))
    current_coverage = 0.0
    current_mask = np.zeros(attr_map.shape, dtype=np.bool)

    if verbose:
        pbar = tqdm.tqdm(total=coverage, desc='Finding attribution masks', bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}]", leave=False)

    max_segment_size = np.prod(input_value.shape) * max_segment_percent
    remaining_masks = {index: mask for index, mask in enumerate(segments)}
    while current_coverage < coverage:
        best_gain = [-np.inf, None]
        remove_queue = []
        inverse_mask = np.logical_not(current_mask)
        for mask_key in remaining_masks:
            mask = np.logical_and(remaining_masks[mask_key], inverse_mask)
            if mask.sum() < min_segment_size or mask.sum() > max_segment_size:
                remove_queue.append(mask_key)
                continue
            remaining_masks[mask_key] = mask
            gain = attr_map[mask].mean()
            if gain > best_gain[0]:
                best_gain = [gain, mask_key]

        for key in remove_queue:
            del remaining_masks[key]
        if len(remaining_masks) == 0:
            break
        added_mask = remaining_masks[best_gain[1]]
        current_mask = np.logical_or(current_mask, added_mask)
        increase_coverage = np.mean(current_mask) - current_coverage
        if verbose:
            pbar.update(increase_coverage)
        current_coverage += increase_coverage
        output_attr[added_mask] = best_gain[0]
        del remaining_masks[best_gain[1]]
    if verbose:
        pbar.update(coverage - current_coverage)
        pbar.close()

    uncovered_mask = output_attr == -np.inf
    smooth_attr = skimage.filters.gaussian(attr_map, sigma=15)
    output_attr[uncovered_mask] = smooth_attr[uncovered_mask]
    if return_raw_map:
        return output_attr, attr_map
    return output_attr
