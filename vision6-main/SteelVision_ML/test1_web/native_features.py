# Extracted verbatim from test1.ipynb. No notebook cells are executed.
import cv2
import numpy as np
from PIL import Image


FEATURE_NAMES = ["mean", "std", "p05", "p25", "p50", "p75", "p95",
                 "row_mean_std", "col_mean_std", "grad_x_mean", "grad_x_p95",
                 "grad_y_mean", "grad_y_p95", "edge_fraction"]


for segment in range(5):
    FEATURE_NAMES += [f"segment_{segment + 1}_{name}" for name in
                      ("mean", "std", "p10", "p90", "edge_fraction")]


def extract_features(path):
    with Image.open(path) as image:
        image.draft("L", (200, 32))
        gray = np.asarray(image.convert("L").resize((200, 32), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    q = np.percentile(gray, [5, 25, 50, 75, 95])
    result = [gray.mean(), gray.std(), *q,
              gray.mean(axis=1).std(), gray.mean(axis=0).std(),
              grad_x.mean(), np.percentile(grad_x, 95),
              grad_y.mean(), np.percentile(grad_y, 95),
              (grad_x > 0.10).mean()]
    for segment in np.array_split(gray, 5, axis=1):
        edge = np.abs(np.diff(segment, axis=1))
        result.extend([segment.mean(), segment.std(),
                       np.percentile(segment, 10), np.percentile(segment, 90),
                       (edge > 0.10).mean()])
    return np.asarray(result, dtype=np.float32)


def cv2_preprocess(path, size, method="raw"):
    width, height = size
    flag = (cv2.IMREAD_GRAYSCALE if width > 400 else
            cv2.IMREAD_REDUCED_GRAYSCALE_4 if width > 200 else
            cv2.IMREAD_REDUCED_GRAYSCALE_8)
    gray = cv2.imread(str(path), flag)
    if gray is None:
        raise FileNotFoundError(path)
    if gray.shape != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    if method == "clahe":
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 2)).apply(gray)
    elif method == "sobel":
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        gray = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif method == "hats":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        gray = cv2.add(bright, dark)
    elif method != "raw":
        raise ValueError(method)
    return gray


def features_from_gray(gray_u8):
    gray = gray_u8.astype(np.float32) / 255.0
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    q = np.percentile(gray, [5, 25, 50, 75, 95])
    result = [gray.mean(), gray.std(), *q,
              gray.mean(axis=1).std(), gray.mean(axis=0).std(),
              grad_x.mean(), np.percentile(grad_x, 95),
              grad_y.mean(), np.percentile(grad_y, 95),
              (grad_x > 0.10).mean()]
    for segment in np.array_split(gray, 5, axis=1):
        edge = np.abs(np.diff(segment, axis=1))
        result.extend([segment.mean(), segment.std(),
                       np.percentile(segment, 10), np.percentile(segment, 90),
                       (edge > 0.10).mean()])
    return np.asarray(result, dtype=np.float32)


def cv2_features(path, size, method="raw"):
    return features_from_gray(cv2_preprocess(path, size, method))


def mask_to_scored_boxes(mask, score_map=None, rank_rule="mean"):
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if score_map is not None:
        if score_map.shape != mask.shape:
            raise ValueError("점수 지도와 마스크의 크기가 다릅니다.")
        score_sums = np.bincount(components.ravel(), weights=score_map.ravel(),
                                 minlength=count)
    boxes = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        score = (float(score_sums[component] / area / 255.0)
                 if score_map is not None else None)
        boxes.append((x, y, width, height, area, score))
    if score_map is None:
        return sorted(boxes, key=lambda box: box[4], reverse=True)
    if rank_rule == "mean":
        return sorted(boxes, key=lambda box: (box[5], box[4]), reverse=True)
    if rank_rule == "sum":
        return sorted(boxes, key=lambda box: (box[4] * box[5], box[5]), reverse=True)
    raise ValueError(f"Unknown box ranking rule: {rank_rule}")
