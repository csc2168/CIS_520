import numpy as np
from PIL import Image
import glob
import os
from sklearn.metrics import jaccard_similarity_score

prediction_folder= "predictions"
gt_image_folder = "annotations"

IOU = []
for prediction_fname in glob.glob(prediction_folder):
    basename = os.path.basename(prediction_fname)
    gt_image_fname = os.path.join(gt_image_folder, basename)

    prediction = np.array(Image.open(prediction_fname))
    gt_im = np.array(Image.open(gt_image_fname))

    IOU.append(jaccard_similarity_score(gt_im, prediction))

print("Meam IOU (i.e. Jaccard Silarity Score):", sum(IOU)/len(IOU))