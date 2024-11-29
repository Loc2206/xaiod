import numpy as np
import cv2
import torch
import yolox.data.data_augment as data_augment
from yolox import models
from yolox.utils import postprocess
from config import device
from data.coco.dataloader import coco_dataloader
from metrics.metric import MetricEvaluation
from xai_methods.dclose import DCLOSE
from xai_methods.drise import DRISE
from xai_methods.gcame import GCAME


def main():
    """
    Example of calculating metrics for XAI methods on YOLOX model with COCO dataset
    :return:
    """
    image_path = 'data/coco/val2017/000000000139.jpg'
    model = models.yolox_l(pretrained=True)
    transform = data_augment.ValTransform(legacy=False)
    org_img = cv2.imread(image_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    h, w, c = org_img.shape
    ratio = min(640 / h, 640 / w)
    img, _ = transform(org_img, None, (640, 640))
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img.requires_grad = False
    model.eval()
    # Get prediction
    with torch.no_grad():
        out = model(img.to(device))
        box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)

    model.zero_grad()
    metric = MetricEvaluation(model, (640, 640))

    idx = 1  # Select the index of the target bounding box
    gcame_method = GCAME(model)
    gcame_map = gcame_method(org_img, box, idx)
    ebpg = metric.ebpg(org_img, gcame_map, box[idx], org_size=None)
    pg = metric.pointing_game(org_img, gcame_map, box[idx])
    drop_conf = metric.drop_conf(org_img, gcame_map, box[idx])
    deletion = metric.del_ins(model, org_img, box, gcame_map, 'del', 1)
    insertion = metric.del_ins(model, org_img, box, gcame_map, 'ins', 1)
    print(f'GCAME - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')

    drise_method = DRISE(model)
    drise_map = drise_method(org_img, box[idx])
    ebpg = metric.ebpg(org_img, drise_map, box[idx], org_size=None)
    pg = metric.pointing_game(org_img, drise_map, box[idx])
    drop_conf = metric.drop_conf(org_img, drise_map, box[idx])
    deletion = metric.del_ins(model, org_img, box, drise_map, 'del', 1)
    insertion = metric.del_ins(model, org_img, box, drise_map, 'ins', 1)
    print(f'DRISE - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')

    dclose_method = DCLOSE(arch='yolox', model=model, img_size=(640, 640), n_samples=1000)
    dclose_map = dclose_method(org_img, box)
    ebpg = metric.ebpg(org_img, dclose_map, box[idx], org_size=None)
    pg = metric.pointing_game(org_img, dclose_map, box[idx])
    drop_conf = metric.drop_conf(org_img, dclose_map, box[idx])
    deletion = metric.del_ins(model, org_img, box, dclose_map, 'del', 1)
    insertion = metric.del_ins(model, org_img, box, dclose_map, 'ins', 1)
    print(f'DCLOSE - EBPG: {ebpg}, PG: {pg}, Drop Conf: {drop_conf}, Deletion: {deletion}, Insertion: {insertion}')
