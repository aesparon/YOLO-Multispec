import numpy as np
import tifffile
from pathlib import Path
from ultralytics.data.utils import verify_image_label as original_verify_image_label
from ultralytics.utils import LOGGER
import cv2


def load_multiband_image(path, channels=5):
    """
    Loads a 5-band (or more) TIFF image using tifffile.
    Normalizes to 0-255 uint8 if needed.
    """
    img = tifffile.imread(path)
    if img.dtype != np.uint8:
        img = ((img / np.iinfo(img.dtype).max) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = img[:, :, None]
    return img[:, :, :channels]  # truncate if more than expected


def patched_load_image(self, i):
    """
    Patch to override YOLODataset.load_image to support 5-band TIFFs.
    """
    img_path = self.im_files[i]
    im = load_multiband_image(img_path, channels=self.data.get("channels", 5))
    return im, im.shape[:2]  # img, (h, w)


def patched_verify_image_label(args):
    """
    Patch to override YOLO label verification with 5-band support.
    Replaces image loader inside verify_image_label.
    """
    im_file, lb_file, prefix, keypoints, nc, nkpt, ndim, single_cls = args
    try:
        im = load_multiband_image(im_file)
        h, w = im.shape[:2]
        shape = (h, w)
        assert im is not None, f"Image Not Found {im_file}"

        if lb_file.exists():
            with open(lb_file, 'r') as f:
                lines = [x.split() for x in f.read().strip().splitlines() if len(x.strip())]
            if any(len(x) < 5 for x in lines):
                raise ValueError(f"Label file {lb_file} has incorrect format.")
            labels = np.array(lines, dtype=np.float32)
            if single_cls:
                labels[:, 0] = 0  # force single-class
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        nl = len(labels)
        if nl:
            cls = labels[:, 0]
            if not ((cls >= 0).all() and (cls < nc).all()):
                raise ValueError(f"Labels in {lb_file} have invalid class indices.")

            # Normalized xywh
            if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
                raise ValueError(f"Labels in {lb_file} are not normalized between 0 and 1.")

        segments = []  # for segmentation, not needed here
        keypoints_data = None
        return im_file, labels, shape, segments, keypoints_data, 0, 1, 0, 0, ''

    except Exception as e:
        return im_file, np.zeros((0, 5), dtype=np.float32), (0, 0), [], None, 0, 0, 0, 1, f"⚠️ {prefix} {im_file}: {e}"


def apply_yolo_multiband_patch():
    """
    Apply monkey-patches to YOLO internals to support multispectral TIFF loading.
    """
    import ultralytics.data.utils
    import ultralytics.data.base

    LOGGER.info("🐍 Applying YOLO multispectral image patch...")

    ultralytics.data.utils.verify_image_label = patched_verify_image_label
    ultralytics.data.base.BaseDataset.load_image = patched_load_image

    LOGGER.info("✅ Patch applied: 5-band TIFF loading enabled.")
