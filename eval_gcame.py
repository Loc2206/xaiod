import numpy as np
import cv2
import torch
import YOLOX.yolox.data.data_augment as data_augment
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
from config import device
from data.coco.dataloader import coco_dataloader
from ebpg import energy_based_pointing_game
from pg import pointing_game
from del_ins import del_ins
from xai_methods.dclose import DCLOSE
from xai_methods.gcame import GCAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare for GCAME output
target_layer = [
    'head.cls_convs.0.0.act'
    'head.cls_convs.0.1.act',
    'head.cls_convs.1.0.act',
    'head.cls_convs.1.1.act',
    'head.cls_convs.2.0.act',
    'head.cls_convs.2.1.act',
]

# Get pretrained model and its transform function
model = models.yolox_l(pretrained=True)
transform = data_augment.ValTransform(legacy=False)

# Read and transform image
img_path = 'data/coco/val2017/000000000139.jpg'

org_img = cv2.imread(img_path)
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
h, w, c = org_img.shape
ratio = min(640 / h, 640 / w)
img, _ = transform(org_img, None, (640, 640))
img = torch.from_numpy(img).unsqueeze(0).float()
img_np = img.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
name_img = img_path.split('/')[-1].split('.')[0]

# GCAME
# Get prediction
img.requires_grad = False
model.eval()
obj_idx = 0

with torch.no_grad():
    out = model(img.to(device))
    box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
    # as there is only 1 input image, box is a tensor with each line representing an object detected in the image -> box[0][1] is the second object detected
    # each line has 7 elements: x1, y1, x2, y2, obj_conf, class_conf, class_ID
    box = box[0]

model.zero_grad()
gcame = GCAME(model, target_layer)
saliency_map = gcame(img.to(device), box=box[obj_idx], obj_idx=obj_idx)
# Expand one dim for saliency map
saliency_map = np.expand_dims(saliency_map, axis=0)

del_auc = np.zeros(80)
ins_auc = np.zeros(80)
count = np.zeros(80)
with torch.no_grad():
    del_img, count_img = del_ins(model, img_np, box, saliency_map, 'del', step=2000)
    # ins_img, count_img = del_ins(model, img_np, box, explanation_map, 'ins', step = 2000)
    del_auc += del_img
    # ins_auc += ins_img
    count += count_img
print("Deletion:", np.mean(del_auc[count != 0] / count[count != 0]))
