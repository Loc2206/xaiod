import torch
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
import YOLOX.yolox.data.data_augment as data_augment
from del_ins import del_ins
from xai_methods.dclose import DCLOSE

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

dclose = DCLOSE(arch="yolox", model=model, img_size=(640, 640), n_samples=4000)
# forward image
obj_idx = 0
with torch.no_grad():
    out = model(img.to(device))
    box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
    box = box[0]
    rs = dclose(img, box, obj_idx)
np.save(f'{name_img}_{obj_idx}.npy', rs)

# create array to save results
del_auc = np.zeros(80)
ins_auc = np.zeros(80)
count = np.zeros(80)

# forward image
with torch.no_grad():
    out = model(img.to(device))
    box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
    box = box[0]
    # if box is None:
    #     continue
    explanation_map = np.load(f'{name_img}_{obj_idx}.npy')
    del_img, count_img = del_ins(model, img_np, box, explanation_map, 'del', step = 2000)
    # ins_img, count_img = del_ins(model, img_np, box, explanation_map, 'ins', step = 2000)
    del_auc += del_img
    # ins_auc += ins_img
    count += count_img
print("Deletion:", np.mean(del_auc[count!=0]/count[count!=0]))
# print("Insertion:", np.mean(ins_auc[count!=0]/count[count!=0]))