import cv2
import numpy as np


def grayscale(image):
    image = np.asarray(image)
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError('Expected nonempty grayscale or BGR image')
    if image.dtype != np.uint8:
        raise ValueError('Expected uint8 input image')
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError('Expected BGR/BGRA image')
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return image.copy()


def alignment_matrix(image):
    gray = grayscale(image)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0.
    if contours:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        angle = float(rect[2])
        if angle > 45:
            angle -= 90
    h, w = gray.shape
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.), angle


def align_image_masks(image, masks):
    matrix, angle = alignment_matrix(image)
    h, w = image.shape[:2]
    aligned = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    transformed = {c: cv2.warpAffine(m, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) for c, m in masks.items()}
    return aligned, transformed, angle


def adaptive_sharpen(image, kernel=5, threshold=10, strength=1.5):
    original = image.astype(np.float32)
    diff = original - cv2.GaussianBlur(original, (kernel, kernel), 0)
    result = original + strength * diff * (np.abs(diff) >= threshold)
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess(image, config=None, return_steps=False):
    cfg = config or {}
    gray = grayscale(image)
    if cfg.get('align', False):
        gray, _, _ = align_image_masks(gray, {})
    steps = {'grayscale': gray}
    current = gray
    if cfg.get('clahe', True):
        current = cv2.createCLAHE(clipLimit=float(cfg.get('clip_limit', 2)), tileGridSize=(8, 8)).apply(current)
        steps['clahe'] = current
    if cfg.get('bilateral', True):
        current = cv2.bilateralFilter(current, 9, 75, 75)
        steps['bilateral'] = current
    if cfg.get('sharpen', True):
        current = adaptive_sharpen(current)
        steps['adaptive_sharpen'] = current
    if cfg.get('opening', True):
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        steps['opening'] = current
    return (current, steps) if return_steps else current


def comparison_variants(image):
    g = grayscale(image)
    f = g.astype(np.float32)
    blur = cv2.GaussianBlur(f, (5, 5), 0)
    clip = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return {
        'normalization_comparison': {'original': g, 'hist_equal': cv2.equalizeHist(g),
            'clahe_2': cv2.createCLAHE(2., (8, 8)).apply(g), 'clahe_4': cv2.createCLAHE(4., (8, 8)).apply(g),
            'min_max': cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX), 'gamma_0.5': clip(255 * (f / 255) ** .5)},
        'filter_comparison': {'original': g, 'average': cv2.blur(g, (5, 5)), 'gaussian': cv2.GaussianBlur(g, (5, 5), 0),
            'median': cv2.medianBlur(g, 5), 'bilateral': cv2.bilateralFilter(g, 9, 75, 75)},
        'sharpening_comparison': {'original': g, 'standard': cv2.filter2D(g, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])),
            'unsharp': clip(f + 1.5 * (f-blur)), 'laplacian': clip(f-cv2.Laplacian(f, cv2.CV_32F)), 'adaptive': adaptive_sharpen(g)},
        'morphology_comparison': {name: cv2.morphologyEx(g, op, kernel) for name, op in
            [('opening', cv2.MORPH_OPEN), ('closing', cv2.MORPH_CLOSE), ('gradient', cv2.MORPH_GRADIENT), ('top_hat', cv2.MORPH_TOPHAT), ('black_hat', cv2.MORPH_BLACKHAT)]}
    }
