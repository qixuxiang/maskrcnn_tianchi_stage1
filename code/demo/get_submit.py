import matplotlib.pyplot as plt
import os
import glob
import cv2
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import json
# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
config_file = "./configs/retinanet/retinanet_R-50-FPN_2x.yaml"
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
infer_path = '../data/First_round_data/jinnan2_round1_test_b_20190326'
# infer_path = '../datasets/jinnan/jinnan2_round1_train_20190305/normal'
images = glob.glob('{}/*.jpg'.format(infer_path))


file_infos = []

for image in images:

    file_info = []
    basename = os.path.basename(image)
    save_path = 'results/' + basename.replace('jpg', 'png')

    image = cv2.imread(image)

    predictions, confidence, boxes, labels = coco_demo.run_on_opencv_image(image)
    print(basename)  # 文件名，如163.jpg

    # cv2.imwrite(save_path, predictions)
    file_info = []
    for i in range(len(boxes)):
        info_dict = {"xmin": int(boxes[i][0]),"xmax": int(boxes[i][2]), "ymin": int(boxes[i][1]), "ymax": int(boxes[i][3]),"label":int(labels[i]),"confidence": round(float(confidence[i]), 3)}
        file_info.append(info_dict)

    file_dict = {'filename':basename, 'rects': file_info}
    file_infos.append(file_dict)

det_path = '../submit/result_submit_b.json'
image_infos = {'results': file_infos}    
fw = open(det_path, 'w')
json.dump(image_infos, fw)