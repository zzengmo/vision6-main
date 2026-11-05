import cv2
import numpy as np


def auto_canny(image, sigma=.33):
    median = np.median(image)
    low = int(max(0, (1 - sigma) * median))
    high = int(min(255, (1 + sigma) * median))
    return cv2.Canny(image, low, max(low + 1, high))


def edge_variants(image):
    return {'canny': cv2.Canny(image, 50, 150), 'auto_canny': auto_canny(image),
            'sobel_x': cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_32F, 1, 0)),
            'sobel_y': cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_32F, 0, 1)),
            'laplacian': cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_32F)),
            'LoG': cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(image, (5, 5), 0), cv2.CV_32F))}


def threshold_variants(image):
    return {'otsu': cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            'fixed': cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1],
            'triangle': cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[1],
            'adaptive_mean': cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
            'adaptive_gaussian': cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)}


def contour_features(image):
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = {'edge_density': float((auto_canny(image) > 0).mean()), 'contour_count': len(contours)}
    rows = []
    for c in contours:
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
        _, _, w, h = cv2.boundingRect(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        rows.append([area, perimeter, w / max(h, 1), 4*np.pi*area/(perimeter**2) if perimeter else 0,
                     area/hull if hull else 0, area/(w*h)])
    stats = np.array(rows) if rows else np.zeros((1, 6))
    result['contour_area_total'] = float(stats[:, 0].sum())
    result['contour_area_max'] = float(stats[:, 0].max())
    for i, name in enumerate(['area', 'perimeter', 'aspect', 'circularity', 'solidity', 'extent']):
        result[f'contour_{name}_mean'] = float(stats[:, i].mean())
    hu = cv2.HuMoments(cv2.moments(binary)).ravel()
    for i, value in enumerate(hu):
        result[f'hu_{i}'] = float(-np.sign(value)*np.log10(abs(value)+1e-30))
    return result
