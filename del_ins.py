import numpy as np
import torch
import cv2
from scipy.spatial import distance
from YOLOX.yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
transform = data_augment.ValTransform(legacy=False)
import math
from torchvision.ops import box_iou
from sklearn.metrics import auc as sklearn_auc
from config import device

def del_ins(model, org_img, box, saliency_map, mode='del', step=100, kernel_width=0.25):
    """
    Calculate the Deletion or Insertion metric for object detection.
    
    Parameters:
        - model: PyTorch nn.Module, the pretrained YOLOX model.
        - org_img: np.ndarray, original input image (H, W, 3).
        - box: torch.Tensor, predicted bounding boxes and scores.
        - saliency_map: np.ndarray, saliency map for each detected object (num_boxes, H, W).
        - mode: str, either 'del' (deletion) or 'ins' (insertion).
        - step: int, number of pixels modified per iteration.
        - kernel_width: float, control parameter for weighting.
    
    Returns:
        - del_ins: np.ndarray, aggregated deletion/insertion scores for each detected class.
        - count: np.ndarray, number of objects per class.
    """
    num_classes = 80  # Assume COCO dataset
    del_ins = np.zeros(num_classes)
    count = np.zeros(num_classes)

    H, W, C = org_img.shape
    HW = H * W
    n_steps = (HW + step - 1) // step  # Number of modification steps

    for idx in range(len(saliency_map)):
        target_cls = int(box[idx][-1])  # Class ID for the current object

        # Initialize start and finish images based on the mode
        if mode == 'del':
            start = org_img.copy()
            finish = np.zeros_like(start)
        else:  # 'ins'
            start = cv2.GaussianBlur(org_img, (51, 51), 0)
            finish = org_img.copy()

        # Flatten the saliency map and sort pixels by importance
        salient_order = np.argsort(-saliency_map[idx].flatten())
        scores = np.zeros(n_steps + 1)

        with torch.no_grad():
            for i in range(n_steps + 1):
                # Get the current set of pixels to modify
                indices = salient_order[step * i: step * (i + 1)]
                y_coords, x_coords = np.unravel_index(indices, (H, W))

                # Modify pixels based on the mode
                start[y_coords, x_coords, :] = finish[y_coords, x_coords, :]

                # Transform the modified image to YOLOX input size
                modified_img, _ = transform(start, None, (640, 640))
                modified_img = torch.from_numpy(modified_img).unsqueeze(0).float().to(device)

                # Forward pass through the model
                output = model(modified_img)
                pred_box, _ = postprocess(output, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
                pred_box = pred_box[0] if pred_box is not None else None

                if pred_box is None:
                    scores[i] = 0
                else:
                    temp_scores = []
                    for det in pred_box:
                        det_cls = int(det[-1])  # Detected class
                        det_box = det[:4]  # Bounding box coordinates
                        det_score = det[5:-1]  # Confidence scores

                        # Calculate IoU and cosine similarity
                        iou = box_iou(det_box.unsqueeze(0), box[idx][:4].unsqueeze(0)).cpu().item()
                        distances = distance.cosine(det_score.cpu(), box[idx][5:-1].cpu())
                        weight = math.exp(-(distances ** 2) / (kernel_width ** 2)) if det_cls == target_cls else 0

                        # Weighted score
                        temp_scores.append(iou * weight)

                    # Use the maximum score for this step
                    scores[i] = max(temp_scores, default=0)

        # Update class-specific metrics
        del_ins[target_cls] += sklearn_auc(scores)
        count[target_cls] += 1

    return del_ins, count
