
# maskrcnn

## 系统环境

### 硬件环境

1. 2080Ti
2. i7-6800K
3. RAM >=32G（训练实机128G）

**注意：** 这是实际在本地训练的硬件配置，不是强制要求硬件配置
### 软件环境

1. cuda 10.1.105
2. cudnn 7.5
3. Pytorch 最新测试预览版本 (注意：既不是1.0正式版也不是1.0.1正式版，最新的预览版本安装见[链接](https://pytorch.org/get-started/locally/))
4. Anaconda 5.0.1| Python 3.6.3

**注意：** 除了Pytorch版本之外，其他的软件版本不是强制要求，只要能成功编译安装[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)即可。
## 安装教程
本项目基于facebook开源的[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)进行修改拓展，安装教程见官方安装指导[INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)。

> 注：安装过后激活虚拟环境命令，官方教程中用的`conda activate maskrcnn_benchmark`可能会报错
>
> 如遇错，改成`source activate maskrcnn_benchmark`即可

### 复现过程

#### 1.数据准备
把比赛所有数据压缩包批量下载并解压放到`data`目录下。`data`目录结构如下（labels中有我们自己随机生成981张normal图片信息的json文件）：

```
|–data
    |-- First_round_data
        |-- jinnan2_round1_train_20190305.zip  # 需解压
        |-- jinnan2_round1_test_a_20190305.zip  # 需解压
        |-- jinnan2_round1_test_b_20190326.zip  # 需解压
        |-- labels
```


如果用我们预训练模型复现的话，先下载[预训练模型](https://pan.baidu.com/s/1Vw22CN0o8PEVbrAJ0__qDA)，提取码xjcv，然后把`model_final.pth`放到`code`目录下，直接执行`python project/code/demo/get_submit.py`即可。

基于maskrcnn制作自己数据集的教程见`code/mask_rcnn_pytorch自定义数据集.md`，这部分无须重复制作，我们已经把这部分嵌入代码工程中。

我们测试时发现基于resnet50的retinanet效果最好，在3月26日更换的第二批测试数据集中，分数达到0.4168。

#### 2.训练复现

训练的时候，第一阶段先用imagenet预训模型训到36000 step，这个阶段不用normal类型图像数据。第一阶段训练结束后，增加981张normal的图像数据后，再加载训练后的模型训练到54000 step。注：这里只加了981张normal的图像数据是因为训练集中restrict类型图像也有981张。

首先执行：
```
python project/code/tools/train_net.py --config-file code/configs/retinanet/retinanet_R-101-FPN_1x.yaml

```
上面命令是先用imagenet预训模型训到36000 step。

然后执行：
```
python project/code/tools/train_net.py --config-file code/configs/retinanet/retinanet_R-101-FPN_2x.yaml
```
上面命令加了981张normal的数据后，再加载训练后的模型训练到54000 step。

详细参数细节可以见两个yaml配置文件。


#### 3.生成提交结果

执行`python project/code/demo/get_submit.py`即可。

如果按前面的步骤放置test集和权重文件，直接执行以下命令：

```shell
python  project/code/demo/get_submit.py
```

运行完毕后可以在`project/submit`路径下找到需要的提交文件`result_submit_b.json`

> 注：如果需要自己指定训练模型和测试图像数据的加载路径，需要修改一些参数
>
> 1、先修改` project/code/configs/retinanet/retinanet_R-101-FPN_2x.yaml`中的WEIGHT字段为你指定的路径
>
> 2、再修改` project/code/demo/get_submit.py`中的`config_file`、`infer_path`和`det_path`变量
>
> 其中：`config_file`是配置文件，`infer_path`是WEIGHT路径，`det_path`是生成提交json结果的路径。
>
> 3、最后执行`python  project/code/demo/get_submit.py`，即可在`project/submit`文件夹中得到提交结果。
