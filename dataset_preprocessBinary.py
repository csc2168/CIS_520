import os
from PIL import Image
import numpy as np
import csv

os.chdir('../FullIJCNN2013')

image_dir = "images"
annotations_dir = "annotations"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
if not os.path.exists(annotations_dir):
    os.mkdir(annotations_dir)
classes = []
with open("gt.txt") as csv_file:
    gt = csv.reader(csv_file, delimiter=';')
    for row in gt:
        im_name, x1, y1, x2, y2, c = row
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        basename = os.path.splitext(im_name)[0]
        classes.append(int(c))
        im = Image.open(im_name)

        gt_im_path = os.path.join(annotations_dir, basename + '.png')
        if os.path.exists(gt_im_path):
            gt_im = np.array(Image.open(gt_im_path))
        else:
            gt_im = np.full(np.array(im).shape[:2], 0, dtype=np.uint8)
        gt_im[y1:y2 + 1, x1:x2 + 1] = 1
        gt_im = Image.fromarray(gt_im)

        im.save(os.path.join(image_dir, basename + ".png"))
        gt_im.save(gt_im_path)
