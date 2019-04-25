import matplotlib.pyplot as plt
import os
import glob
import cv2
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
config_file = "./configs/retinanet/retinanet_R-50-FPN_1x.yaml"
if not os.path.exists('results'):
    os.makedirs('results')

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

# infer_path = '../data/jinnan2_round1_test_a_20190306'
infer_path = '../data/jinnan2_round1_train_20190305/normal'
images = glob.glob('{}/*.jpg'.format(infer_path))

for image in images:
    basename = os.path.basename(image)
    save_path = 'results/' + basename.replace('jpg', 'png')

    image = cv2.imread(image)
    print(basename)

    predictions, confidence, boxes, labels = coco_demo.run_on_opencv_image(image)
    cv2.imwrite(save_path, predictions)

    # break