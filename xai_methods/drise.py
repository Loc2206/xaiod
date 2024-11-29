import cv2
import math
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from utils import *


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class DRISE(object):
    def __init__(self, model, img_size=(608, 608), grid_size=(16, 16), n_samples=1000, prob_thresh=0.2, seed=0,
                 device='cpu', **kwargs):
        """
        Parameters:
          - model: The model in nn.Modules() to analyze
          - img_size: The image size in tuple (H, W)
          - grid_size: The grid size in tuple (h, w)
          - n_samples: Number of samples to create
          - prob_thresh: The appearence probability of 1 grid
        """
        self.model = model.eval()
        self.img_size = img_size

        self.grid_size = grid_size
        self.n_samples = n_samples
        self.prob_thresh = prob_thresh
        self.seed = seed
        self.device = device

    def __call__(self, image, box):
        return self.generate_saliency_map(image, box)

    def generate_mask(self, ):
        """
        Return a mask with shape [H, W]
        """
        image_h, image_w = self.img_size
        grid_h, grid_w = self.grid_size

        # Create cell for mask
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

        # Create {0, 1} mask
        mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
                self.prob_thresh).astype(np.float32)
        # Up-size to get value in [0, 1]
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        # Randomly crop the mask
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)

        mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
        return mask

    def mask_image(self, image, mask):
        """
        Return a masked image with [0, 1] mask
        """
        masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
                  255).astype(np.uint8)
        return masked

    def generate_saliency_map(self, img, box):
        transform = T.Compose([T.ToTensor()])
        np.random.seed(self.seed)
        h, w, c = img.shape
        self.img_size = (h, w)
        saliency_map = np.zeros((h, w), dtype=np.float32)
        target_class = box[2]
        target_score = box[3]
        target_box = list(box[0]) + list(box[1])
        count = 0
        for _ in tqdm(range(self.n_samples)):
            # Create n_samples
            count += 1
            mask = self.generate_mask()
            masked = self.mask_image(img, mask)
            masked = transform(masked)
            masked = masked.unsqueeze(0)

            ious = []
            all_scores_map = []
            self.model.zero_grad()
            with torch.no_grad():
                prediction = self.model(masked.to(device))
            pred_class = list(prediction[0]['labels'].cpu().numpy())
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].cpu().detach().numpy())]
            pred_score = list(prediction[0]['scores'].cpu().detach().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x > 0.5]
            if len(pred_t) == 0:
                continue

            pred_t = pred_t[-1]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]
            scores = pred_score[:pred_t + 1]
            for b in range(len(pred_boxes)):
                if pred_class[b] != target_class:
                    continue
                else:
                    new_bbox = list(pred_boxes[b][0]) + list(pred_boxes[b][1])

                    # cosine = (target_score * scores[b]) / math.sqrt(pow(target_score, 2) + pow(scores[b], 2))
                    iou = bbox_iou(new_bbox, target_box)
                    ious.append(iou)
                    all_scores_map.append(scores[b])

            if len(ious) == 0:
                continue
            t = mask * np.max(ious) * all_scores_map[np.argmax(ious)]
            saliency_map += t
            M, m = saliency_map.max(), saliency_map.min()
            saliency_map = (saliency_map - m) / (M - m)

        return saliency_map


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()

    img = Image.open('data/000000008021.jpg')
    img = np.array(img)
    box = [(0, 0), (100, 100), 1, 0.9]

    drise = DRISE(model, device=device)
    saliency_map = drise(img, box)
