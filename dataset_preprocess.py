import os
from PIL import Image
import numpy as np
import csv

os.chdir('../FullIJCNN2013')

if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("annotations"):
    os.mkdir("annotations")
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
        gt_im = np.full(np.array(im).shape[:2], 43, dtype=np.uint8) #Set background to max class + 1
        gt_im[y1:y2+1, x1:x2+1] = c
        gt_im = Image.fromarray(gt_im)
        
        im.save(os.path.join("images", basename + ".png"))
        gt_im.save(os.path.join("annotations", basename + ".png"))
