import os
from PIL import Image
import numpy as np
import csv


def read_gts(filename):

	if not os.path.exists("png_images"):
		os.mkdir("png_images")
	if not os.path.exists("gt_images"):
		os.mkdir("gt_images")

	if filename > 9:
		str1 = '/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Training/Images/000%d' % (filename)
		str2 = "GT-000%d.csv" % (filename)
	else:
		str1 = '/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Training/Images/0000%d' % (filename)
		str2 = "GT-0000%d.csv" % (filename)
		
	os.chdir(str1)

	classes = []
	with open(str2) as csv_file:
		gt = csv.reader(csv_file, delimiter=';')
		next(gt)
		for row in gt:
			im_name, w, h, x1, y1, x2, y2, c = row
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
			
			im.save(os.path.join("../../../../png_images", basename + ".png"))
			gt_im.save(os.path.join("../../../../gt_images", basename + ".png"))

for i in range(0,42): 
	read_gts(i)