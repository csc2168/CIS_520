from __future__ import division
import numpy as np
from PIL import Image
import glob
import os
from collections import defaultdict
from matplotlib import pyplot as plt

prediction_folder = r"Predictions\BinaryPredictions\predictions"
gt_image_folder = r"Predictions\BinaryPredictions\annotations"
prediction_folder = r"CNN_full_predicts"
gt_image_folder = r"Predictions\classesPrediction\annotations"
graph_folder="CNN_graphs"
# prediction_folder = r"..\logsBReweight\predictions"
# gt_image_folder = r"..\FullIJCNN2013\annotations\testing"


if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)
IOU = []
IOU_per_class = defaultdict(list)

for gt_image_fname in glob.glob(os.path.join(gt_image_folder, "*.png")):
    basename = os.path.basename(gt_image_fname)
    prediction_fname = os.path.join(prediction_folder, "predict_" + basename)
    if not os.path.exists(prediction_fname):
        continue
    prediction = np.array(Image.open(prediction_fname))
    gt_im = np.array(Image.open(gt_image_fname))

    im_iou = []
    for val in np.unique([prediction, gt_im]):
        pred_vals = prediction == val
        gt_vals = gt_im == val
        intersection = pred_vals & gt_vals
        union = pred_vals | gt_vals
        calculated_iou = intersection.sum()/union.sum()
        im_iou.append(calculated_iou)
        IOU_per_class[val].append(calculated_iou)

    IOU.append(sum(im_iou)/len(im_iou))

mean_IOU_per_class = {c: sum(vs)/len(vs) for c, vs in IOU_per_class.items()}
print("Meam IOU (i.e. Jaccard Similarity Score):", sum(IOU)/len(IOU))
print(mean_IOU_per_class)


plt.hist(IOU)
plt.title("Mean IOU")
plt.savefig(os.path.join(graph_folder, "mean_iou.png"))
plt.show()

for (c, ious) in IOU_per_class.items():
    #plt.figure()
    fig, ax = plt.subplots()
    plt.hist(ious)
    plt.title("Class {} IOU".format(c))
    plt.savefig(os.path.join(graph_folder, "{}_iou.png".format(c)))
