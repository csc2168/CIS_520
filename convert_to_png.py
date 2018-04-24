import os
from PIL import Image
import numpy as np
import csv
import glob

os.chdir('FullIJCNN2013')
if not os.path.exists("png_images"):
    os.mkdir("png_images")

for im_name in glob.glob("*.ppm"):
    basename = os.path.splitext(im_name)[0]
    im = Image.open(im_name)
    im.save(os.path.join("png_images", basename + ".png"))
    
    
