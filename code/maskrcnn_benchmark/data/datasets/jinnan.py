from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
import os
import json
import torch

class MyDataset(object):
    def __init__(self,ann_file=None, root=None, remove_images_without_annotations=None, transforms=None):
        # as you would do normally

        self.transforms = transforms
        
        #normal or restricted
        self.is_normal = False

        self.train_path = root
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        
        if ann_file.split('/')[-1] == "normal_no_poly.json":
            self.is_normal = True

        # 看要训练的图像有多少张，把id用个列表存储方便随机
        self.idxs = list(range(len(self.data['images'])))
        self.bbox_label = {}

        if not self.is_normal:
            for anno in self.data['annotations']:
                bbox = anno['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                cate = anno['category_id']
                image_id = anno['image_id']
                if not image_id in self.bbox_label:
                    self.bbox_label[image_id] = [[bbox], [cate]]
                else:
                    self.bbox_label[image_id][0].append(bbox)
                    self.bbox_label[image_id][1].append(cate)

    def __getitem__(self, idx):
        
            # load the image as a PIL Image
        idx = self.idxs[idx % len(self.data['images'])]

        #normal without boxes and lables
        if self.is_normal:
            folder = 'normal'
            boxes = [[0, 0, 0, 0]] 
            category =  [0] 
        else:
            folder = 'restricted'
            # x1, y1, x2, y2 order.
            # has bbox and labels
            if idx in self.bbox_label:  
                boxes = self.bbox_label[idx][0]
                category = self.bbox_label[idx][-1]
            else:
                # hasnot bbox and labels
                boxes = [[0, 0, 0, 0]]
                category =  [0] 

        path = self.data['images'][idx]['file_name']

        image = Image.open(os.path.join(self.train_path, folder, path)).convert('RGB')
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        labels = torch.tensor(category)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)
   
        # return the image, the boxlist and the idx in your dataset
        
        return image, boxlist, idx

    def __len__(self):
        return len(self.data['images'])

    def get_img_info(self, idx):
        idx = self.idxs[idx % len(self.data['images'])]
        height = self.data['images'][idx]['height']
        width = self.data['images'][idx]['width']
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": height, "width": width}