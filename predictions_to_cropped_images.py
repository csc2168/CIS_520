import glob
import os
from skimage import measure, io
from skimage.transform import resize
import re


predictions_folder = "Predictions/BinaryPredictions/predictions"
orig_image_folder = "FullIJCNN2013/png_images"
cropped_image_folder = "bin_predicted_cropped_images"

if not os.path.exists(cropped_image_folder):
    os.makedirs(cropped_image_folder)

re_identifier = re.compile(r'predict_(\d*).png')
for full_predict_fname in glob.glob(os.path.join(predictions_folder, "*.png")):
    full_predict = io.imread(full_predict_fname)
    basename = os.path.basename(full_predict_fname)
    identifier = re.match(re_identifier, basename).group(1)
    full_image = io.imread(os.path.join(orig_image_folder, identifier + ".png"))

    #Convert full_predict to binary if not already
    #Can be removed if binary images are used
    max_full_predict = full_predict.max()
    if max_full_predict > 1:
        full_predict[full_predict < max_full_predict] = 1
        full_predict[full_predict == max_full_predict] = 0

    for i, region in enumerate(measure.regionprops(full_predict)):
        min_row, min_col, max_row, max_col = region.bbox
        cropped_im = full_image[min_row:max_row, min_col:max_col]
        cropped_im_name = "{}_{}.png".format(identifier, i)
        io.imsave(os.path.join(cropped_image_folder, cropped_im_name), cropped_im)


