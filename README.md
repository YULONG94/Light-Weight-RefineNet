# Light-Weight-RefineNet
pytorch win10

单个GTX 1060的batch size可以设置为4， 单个GTX 1070可以设置成6，如果你使用的是非常高级的装备，当然调高一点更好，目前还在跑模型的过程中，稍后上传

## 训练数据集下载
我使用的是增强版的VOC2012，用于做语义识别，图像分割的共有12031张原图和标签，可在百度网盘的链接下载：

下载解压可以得到如下文件（夹）

JPEGImages：文件夹，包含17125张图片，其中12031张用于这个语义分割的任务

SegmentationClassAug：文件夹，包含12031张标签图片（需要是灰度图格式）

test.txt

train.txt

train_aug.txt

trainval.txt

trainval_aug.txt

val.txt

## 训练
configs.py和utils中的get_arguments函数支持多种参数的设置，除了需要更改关于训练数据的文件位置，其他都接受默认的配置即可运行

>TRAIN_DIR改成存放VOC2012AUG（上方的数据集下载可以获得）文件夹的路径即可

>同理，对于TRAIN_LIST和VAL_LIST

> 我的邮箱联系方式1286307712@qq.com，欢迎联系，如果觉得还不错的话，可以右上角点点星星，谢谢 (^.^)
