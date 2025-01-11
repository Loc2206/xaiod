import numpy as np
import cv2
import torch
import YOLOX.yolox.data.data_augment as data_augment
from YOLOX.yolox import models
from YOLOX.yolox.utils import postprocess
from config import device
from data.coco.dataloader import coco_dataloader
# from metrics.metric import MetricEvaluation
from ebpg import energy_based_pointing_game
from pg import pointing_game
from del_ins import del_ins
from xai_methods.dclose import DCLOSE
from xai_methods.drise import DRISE
from xai_methods.gcame import GCAME

# Get pretrained model and its transform function
model = models.yolox_l(pretrained=True)
transform = data_augment.ValTransform(legacy=False)

# Read and transform image
image_path = 'data/coco/val2017/000000000139.jpg'
org_img = cv2.imread(image_path)
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
h, w, c = org_img.shape
ratio = min(640 / h, 640 / w)
img, _ = transform(org_img, None, (640, 640))
img = torch.from_numpy(img).unsqueeze(0)
img = img.float()

# GCAME
# Get prediction
img.requires_grad = False
model.eval()
with torch.no_grad():
    out = model(img.to(device))
    box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True) 
    # as there is only 1 input image, box is a tensor with each line representing an object detected in the image -> box[0][1] is the second object detected
    # each line has 7 elements: x1, y1, x2, y2, obj_conf, class_conf, class_ID
    
# Prepare for GCAME output
target_layer = [
    'head.cls_convs.0.0.act'
    'head.cls_convs.0.1.act',

    'head.cls_convs.1.0.act',
    'head.cls_convs.1.1.act',

    'head.cls_convs.2.0.act',
    'head.cls_convs.2.1.act',
  ]
model.zero_grad()
idx = 1  # Select the index of the target bounding box
gcame_method = GCAME(model, target_layer)
gcame_map = gcame_method(img.to(device), box=box[0][idx], obj_idx=idx)
print(gcame_map.shape) # ensure size of saliency map is the same as the input image (640, 640)
print(box[0][idx].shape)
# print(box)
# print(box[0][idx])

# Evaluate metrics
# metric = MetricEvaluation(model, (640, 640))
gcame_ebpg = energy_based_pointing_game(gcame_map, box[0]) # input: saliency map, bbox: A tensor of shape (N, 7) for a single image
print(f'GCAME - EBPG: {gcame_ebpg}') # print the EBPG score for the saliency map (all objects detected in the image)
gcame_pg = pointing_game(gcame_map, box[0])
print(f'GCAME - PG: {gcame_pg}') # print the PG score for the saliency map (all objects detected in the image)
gcame_deletion = del_ins(model, org_img, box[0], gcame_map, 'del', 1)
print(f'GCAME - DELETION: {gcame_deletion}') # print the DELETION score for the saliency map (all objects detected in the image)
# drop_conf = metric.drop_conf(org_img, gcame_map, box[idx])
# deletion = metric.del_ins(model, org_img, box, gcame_map, 'del', 1)
# insertion = metric.del_ins(model, org_img, box, gcame_map, 'ins', 1)
# print(f'GCAME - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')

# # D-CLOSE
# dclose_method = DCLOSE(arch='yolox', model=model, img_size=(640, 640), n_samples=1000)
# dclose_map = dclose_method(org_img, box)
# ebpg = metric.ebpg(org_img, dclose_map, box[idx], org_size=None)
# pg = metric.pointing_game(org_img, dclose_map, box[idx])
# drop_conf = metric.drop_conf(org_img, dclose_map, box[idx])
# deletion = metric.del_ins(model, org_img, box, dclose_map, 'del', 1)
# insertion = metric.del_ins(model, org_img, box, dclose_map, 'ins', 1)
# print(f'DCLOSE - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')


# def main():
#     """
#     Example of calculating metrics for XAI methods on YOLOX model with COCO dataset
#     :return:
#     """
#     image_path = 'data/coco/val2017/000000000139.jpg'
#     model = models.yolox_l(pretrained=True)
#     transform = data_augment.ValTransform(legacy=False)
#     org_img = cv2.imread(image_path)
#     org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
#     h, w, c = org_img.shape
#     ratio = min(640 / h, 640 / w)
#     img, _ = transform(org_img, None, (640, 640))
#     img = torch.from_numpy(img).unsqueeze(0)
#     img = img.float()
#     img.requires_grad = False
#     model.eval()
#     # Get prediction
#     with torch.no_grad():
#         out = model(img.to(device))
#         box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)

#     model.zero_grad()
#     metric = MetricEvaluation(model, (640, 640))

#     idx = 1  # Select the index of the target bounding box
#     gcame_method = GCAME(model)
#     gcame_map = gcame_method(org_img, box, idx)
#     ebpg = metric.ebpg(org_img, gcame_map, box[idx], org_size=None)
#     pg = metric.pointing_game(org_img, gcame_map, box[idx])
#     drop_conf = metric.drop_conf(org_img, gcame_map, box[idx])
#     deletion = metric.del_ins(model, org_img, box, gcame_map, 'del', 1)
#     insertion = metric.del_ins(model, org_img, box, gcame_map, 'ins', 1)
#     print(f'GCAME - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')

#     drise_method = DRISE(model)
#     drise_map = drise_method(org_img, box[idx])
#     ebpg = metric.ebpg(org_img, drise_map, box[idx], org_size=None)
#     pg = metric.pointing_game(org_img, drise_map, box[idx])
#     drop_conf = metric.drop_conf(org_img, drise_map, box[idx])
#     deletion = metric.del_ins(model, org_img, box, drise_map, 'del', 1)
#     insertion = metric.del_ins(model, org_img, box, drise_map, 'ins', 1)
#     print(f'DRISE - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')

#     dclose_method = DCLOSE(arch='yolox', model=model, img_size=(640, 640), n_samples=1000)
#     dclose_map = dclose_method(org_img, box)
#     ebpg = metric.ebpg(org_img, dclose_map, box[idx], org_size=None)
#     pg = metric.pointing_game(org_img, dclose_map, box[idx])
#     drop_conf = metric.drop_conf(org_img, dclose_map, box[idx])
#     deletion = metric.del_ins(model, org_img, box, dclose_map, 'del', 1)
#     insertion = metric.del_ins(model, org_img, box, dclose_map, 'ins', 1)
#     print(f'DCLOSE - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')
