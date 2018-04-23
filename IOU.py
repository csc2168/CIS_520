import numpy as np
from PIL import Image
import glob
import os
from sklearn.metrics import jaccard_similarity_score
print(os.getcwd())
prediction_folder = r"D:\CloudStation\PENN\cis520\Project\logs\predictions"
gt_image_folder = r"D:\CloudStation\PENN\cis520\Project\FullIJCNN2013\annotationsClassOld\testing"

IOU = []
for gt_image_fname in glob.glob(os.path.join(gt_image_folder, "*.png")):
    basename = os.path.basename(gt_image_fname)
    prediction_fname = os.path.join(prediction_folder, "predict_" + basename)

    prediction = np.array(Image.open(prediction_fname))
    gt_im = np.array(Image.open(gt_image_fname))

    IOU.append(jaccard_similarity_score(gt_im, prediction))

print("Meam IOU (i.e. Jaccard Similarity Score):", sum(IOU)/len(IOU))