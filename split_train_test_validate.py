import os
import random
image_dir = "../TrainIJCNN2013/images"
annotation_dir = "../TrainIJCNN2013/annotations"
annotation_prefix = ""
images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
random.shuffle(images)
train_test_split_ratio = 0.8
train_val_split_ratio = 0.8
train_val = images[:int(train_test_split_ratio*len(images))]
test = images[int(train_test_split_ratio*len(images)):]
train =  train_val[:int(train_val_split_ratio*len(train_val))]
validation = train_val[int(train_val_split_ratio*len(train_val)):]

for set, fname in zip([train, test, validation], ["training", "testing", "validation"]):
    for f in set:
        os.renames(os.path.join(image_dir, f), os.path.join(image_dir, fname, os.path.basename(f)))
        os.renames(os.path.join(annotation_dir, annotation_prefix + f), os.path.join(annotation_dir, fname, annotation_prefix + os.path.basename(f)))
