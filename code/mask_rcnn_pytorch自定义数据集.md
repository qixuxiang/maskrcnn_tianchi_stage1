# 自定义数据集的方法

---



# 1、拷贝数据集到根目录的datasets下(和demo同级目录)如

```shell
maskrcnn-benchmark/datasets/jinnan/jinnan2_round1_train_20190305
```



# 2、修改paths_catalog.py

路径为`maskrcnn-benchmark/maskrcnn_benchmark/config/paths_catalog.py`

## a、在paths_catalog中的`DATASETS`字典中添加你需要的路径，如

```shell
"jinnan_train": {
"img_dir": "jinnan2_round1_train_20190305",
"ann_file": "jinnan2_round1_train_20190305/train_no_poly.json"
},
```

注意：自定义数据集的话，`img_dir`和`ann_file`会作为形参传到你自己创建的`MyDataset`类里面

## b、修改paths_catalog中部静态函数get(name)方法

添加一个if else，把你创建的数据集相关内容放进去,如

```python
elif "jinnan" in name:  # name对应yaml文件传过来的数据集名字
    data_dir = DatasetCatalog.DATA_DIR
    attrs = DatasetCatalog.DATASETS[name]
    args = dict(
        root=os.path.join(data_dir, attrs["img_dir"]),  # img_dir就是a步骤里面的内容
        ann_file=os.path.join(data_dir, attrs["ann_file"]),  # ann_file就是a步骤里面的内容
    )
    return dict(
        factory="MyDataset",  # 这个MyDataset对应
        args=args,
    )
```

**上面参数解释（主要是`MyDataset`）：**

1) 这个`MyDataset`就是你自己建的那个类，返回值是`image, boxlist, idx`，具体实现参考[git官网](https://github.com/facebookresearch/maskrcnn-benchmark#adding-your-own-dataset)（很容易）

2) 比如我实现好了`MyDataset`类，然后这个`py`文件取名为`jinnan.py`

3) 然后放在`maskrcnn-benchmark/maskrcnn_benchmark/data/datasets`路径下

4) 接着配置那个目录里面的`__init__.py`文件，第四行和all最后一个元素是自己加的

```python
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .jinnan import MyDataset

all = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "MyDataset"]
```

5) 注意，实现MyDataset要实现`__len__`，`__getitem__`，`get_img_info`，还有`__init__`，其中`__init__`会得到第一个步骤传来的`attrs`，`__init__`的一个参数参考：

```python
def __init__(self,ann_file=None, root=None, remove_images_without_annotations=None, transforms=None)
```

不知参数是什么意思得去看`maskrcnn-benchmark/maskrcnn_benchmark/data/build.py`



# 3、修改yaml文件

主要是修改数据load部分

```python
MODEL:
  MASK_ON: False
DATASETS:
  TRAIN: ("jinnan_train", "jinnan_val")
  TEST: ("jinnan_test",)
```

上面三个值都是自己设的，其实有用的就`jinnan_train`，当然首先重要的是要把`MASK_ON`关闭。



# 4、 我自己写的数据加载的参考

`maskrcnn-benchmark/maskrcnn_benchmark/data/datasets/jinnan.py`

```python
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
import os
import json
import torch

class MyDataset(object):
    def __init__(self,ann_file=None, root=None, remove_images_without_annotations=None, transforms=None):
        # as you would do normally

        self.transforms = transforms

        self.train_path = root
        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self.idxs = list(range(len(self.data['images'])))  # 看要训练的图像有多少张，把id用个列表存储方便随机
        self.bbox_label = {}
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
        if idx not in self.bbox_label:  # 210, 262, 690, 855 have no bbox
            idx += 1
        path = self.data['images'][idx]['file_name']

        folder = 'restricted' if idx < 981 else 'normal'

        image = Image.open(os.path.join(self.train_path, folder, path)).convert('RGB')
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        boxes = self.bbox_label[idx][0]
        category = self.bbox_label[idx][-1]

        # and labels
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
```



