import numpy as np
import os
from PIL import Image
import glob

zero_one_folder = "predictions"
visible_folder = "visible_predictions"

for fname in glob.glob(os.path.join(zero_one_folder, "*.png")):
    im = np.array(Image.open(fname))
    im[im == 1] = 255
    out_im = Image.fromarray(im)
    out_im.save(os.path.join(visible_folder, os.path.basename(fname)))
