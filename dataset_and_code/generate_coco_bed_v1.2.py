#!/usr/bin/env python3
"""
generate_coco_bed.py
Create COCO-BED: 3 random background boxes per COCO val image.
Run:
    python generate_coco_bed.py \
        --coco_path /path/to/coco2017 \
        --save_dir ./coco-bed \
        --seed 42
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from typing import List, Tuple

# ---------- 工具 ----------
def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """box: [x, y, w, h] -> compute IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)

def sample_size(ratio_small: float,
                ratio_medium: float) -> Tuple[int, int]:
    p = np.random.rand()
    if p < ratio_small:
        w = np.random.randint(8, 32)
        h = np.random.randint(8, 32)
    elif p < ratio_small + ratio_medium:
        w = np.random.randint(32, 96)
        h = np.random.randint(32, 96)
    else:
        w = np.random.randint(96, 256)
        h = np.random.randint(96, 256)
    return int(w), int(h)

def valid_bg_box(img_w: int, img_h: int,
                 gt_boxes: List[List[float]],
                 cluster_centers: np.ndarray,
                 ratio_small: float,
                 ratio_medium: float,
                 min_iou: float,
                 tries: int = 100):
    for _ in range(tries):
        cc = cluster_centers[np.random.randint(0, len(cluster_centers))]
        cx = np.random.normal(cc[0], img_w * 0.2)
        cy = np.random.normal(cc[1], img_h * 0.2)
        w, h = sample_size(ratio_small, ratio_medium)
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            continue
        bg_box = [x, y, w, h]
        if all(iou(bg_box, gt) < min_iou for gt in gt_boxes):
            return bg_box
    return None

# ---------- 主流程 ----------
def generate_coco_bed(coco_path: str,
                      save_dir: str,
                      n_bg_per_img: int = 3,
                      min_iou: float = 0.3,
                      seed: int = 42,
                      vis_num: int = 20):
    np.random.seed(seed)

    ann_file = os.path.join(coco_path, 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(coco_path, 'images', 'val2017')
    os.makedirs(save_dir, exist_ok=True)

    coco = COCO(ann_file)

    # 直接过滤掉 list 为空的 img_id
    img_ids = [i for i in coco.getImgIds() if coco.loadImgs(i)]
    print(f'有效 img_ids: {len(img_ids)}')
    if not img_ids:
        raise RuntimeError('未找到任何有效 val2017 图片，请检查 --coco_path。')

    # 统计尺寸
    areas, centers = [], []
    for ann in coco.loadAnns(coco.getAnnIds()):
        x, y, w, h = ann['bbox']
        areas.append(w * h)
        centers.append([x + w / 2, y + h / 2])
    centers = np.array(centers)

    kmeans = KMeans(n_clusters=10, random_state=seed).fit(centers)
    cluster_centers = kmeans.cluster_centers_

    areas = np.array(areas)
    ratio_small = (areas < (32 * 32)).mean()
    ratio_medium = ((areas >= (32 * 32)) & (areas < (96 * 96))).mean()

    new_anns, bg_ann_id = [], 1
    for img_id in tqdm(img_ids, desc='Generating COCO-BED'):
        img_info_list = coco.loadImgs(img_id)
        if not img_info_list:          # 再次防御
            continue
        img_info = img_info_list[0]

        img_w, img_h = img_info['width'], img_info['height']
        gt_boxes = [ann['bbox']
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]

        for _ in range(n_bg_per_img):
            box = valid_bg_box(img_w, img_h, gt_boxes,
                               cluster_centers,
                               ratio_small, ratio_medium,
                               min_iou)
            if box is None:
                continue
            x, y, w, h = box
            new_anns.append({
                "id": bg_ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            bg_ann_id += 1

    # 保存
    coco.dataset['annotations'].extend(new_anns)
    coco.dataset['categories'].append({
        "id": 0,
        "name": "background",
        "supercategory": "background"
    })
    save_ann = os.path.join(save_dir, 'instances_val2017_bed.json')
    with open(save_ann, 'w') as f:
        json.dump(coco.dataset, f)
    print(f'Annotations -> {save_ann}')

    # 建立类别 id->name 映射
    cat_id2name = {c['id']: c['name'] for c in coco.dataset['categories']}
    cat_id2name[0] = 'background'        # 我们新增的类别

    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    sample_imgs = np.random.choice(img_ids,
                                   min(vis_num, len(img_ids)),
                                   replace=False)

    for idx, img_id in enumerate(sample_imgs):
        tmp = coco.loadImgs(int(img_id))
        if tmp is None or len(tmp) == 0:
            continue
        img_info = tmp[0]

        img_path = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 画 GT
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cat_name = cat_id2name.get(ann['category_id'], 'unknown')
            cv2.putText(img, cat_name, (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 画背景框
        for ann in [a for a in new_anns if a['image_id'] == img_id]:
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cat_name = cat_id2name.get(ann['category_id'], 'unknown')
            cv2.putText(img, cat_name, (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(vis_dir, f'vis_{idx}.jpg'), img)
    print(f'Visualization (with labels) -> {vis_dir}')
# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./coco-bed')
    parser.add_argument('--n_bg_per_img', type=int, default=3)
    parser.add_argument('--min_iou', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_coco_bed(args.coco_path,
                      args.save_dir,
                      args.n_bg_per_img,
                      args.min_iou,
                      args.seed)
