import cv2
import numpy as np
from .edge_contour import auto_canny


def analyze_roi(image, mask=None, quantile=.75):
    edges = auto_canny(image)
    h, w = image.shape[:2]
    ys, xs = np.linspace(0, h, 5, dtype=int), np.linspace(0, w, 5, dtype=int)
    density = np.zeros((4, 4), np.float32)
    boxes = []
    for row in range(4):
        for col in range(4):
            tile = edges[ys[row]:ys[row+1], xs[col]:xs[col+1]]
            density[row, col] = (tile > 0).mean() if tile.size else 0
    threshold = np.quantile(density, quantile)
    candidate = np.zeros((h, w), bool)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for row in range(4):
        for col in range(4):
            x, y, x2, y2 = int(xs[col]), int(ys[row]), int(xs[col+1]), int(ys[row+1])
            cv2.rectangle(overlay, (x, y), (max(x, x2-1), max(y, y2-1)), (100, 100, 100), 1)
            if density[row, col] > 0 and density[row, col] >= threshold:
                boxes.append([x, y, x2, y2])
                candidate[y:y2, x:x2] = True
                cv2.rectangle(overlay, (x, y), (x2-1, y2-1), (0, 0, 255), 2)
    result = {'edges': edges, 'density': density, 'overlay': overlay, 'boxes': boxes, 'candidate': candidate}
    if mask is not None:
        truth = mask.astype(bool)
        union = (truth | candidate).sum()
        result['iou'] = float((truth & candidate).sum()/union) if union else 1.
        result['defect_coverage'] = float((truth & candidate).sum()/truth.sum()) if truth.any() else None
    return result
