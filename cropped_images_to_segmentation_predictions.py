import glob
import os
from skimage import measure, io
from skimage.transform import resize
import re
import csv


predictions_folder = "Predictions/BinaryPredictions/predictions"
orig_image_folder = "FullIJCNN2013"
cropped_image_folder = "bin_predicted_cropped_images"
class_prediction_file = 'cnn_results.csv'
out_multiclass_folder = "CNN_full_predicts"

if not os.path.exists(out_multiclass_folder):
    os.makedirs(out_multiclass_folder)

with open(class_prediction_file, "rb") as f:
    csvfile = csv.reader(f)
    cropped_im_class_dict = {line[0]: line[1] for line in csvfile}

if not os.path.exists(cropped_image_folder):
    os.makedirs(cropped_image_folder)

re_identifier = re.compile(r'predict_(\d*).png')
for full_predict_fname in glob.glob(os.path.join(predictions_folder, "*.png")):
    full_predict = io.imread(full_predict_fname)
    basename = os.path.basename(full_predict_fname)
    identifier = re.match(re_identifier, basename).group(1)
    full_image = io.imread(os.path.join(orig_image_folder, identifier + ".ppm"))

    #Convert full_predict to binary if not already
    #Can be removed if binary images are used
    max_full_predict = full_predict.max()
    if max_full_predict > 1:
        full_predict[full_predict < max_full_predict] = 1
        full_predict[full_predict == max_full_predict] = 0

    for i, region in enumerate(measure.regionprops(full_predict)):
        min_row, min_col, max_row, max_col = region.bbox
        # cropped_im = full_image[min_row:max_row, min_col:max_col]
        # cropped_im = resize(cropped_im, cropped_size)
        cropped_im_name = "{}_{}.png".format(identifier, i)
        cropped_im = io.imread(os.path.join(cropped_image_folder, cropped_im_name))
        full_predict[region.coords[:, 0], region.coords[:, 1]] = cropped_im_class_dict[cropped_im_name]
    io.imsave(os.path.join(out_multiclass_folder, basename), full_predict)


