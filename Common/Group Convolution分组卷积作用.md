# [Group Convolution分组卷积，以及Depthwise Convolution和Global Depthwise Convolution](https://www.cnblogs.com/shine-lee/p/10243114.html)

**Group Convolution分组卷积**，最早见于AlexNet——2012年Imagenet的冠军方法，Group Convolution被用来切分网络，使其在2个GPU上并行运行，AlexNet网络结构如下：

![AlexNet](https://s2.ax1x.com/2019/01/08/FLPm1P.png)

# Convolution VS Group Convolution

在介绍**Group Convolution**前，先回顾下**常规卷积**是怎么做的，具体可以参见博文[《卷积神经网络之卷积计算、作用与思想》](https://blog.shinelee.me/2018/11-08-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B9%8B%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E3%80%81%E4%BD%9C%E7%94%A8%E4%B8%8E%E6%80%9D%E6%83%B3.html)。如果输入feature map尺寸为C∗H∗W

，卷积核有N个，输出feature map与卷积核的数量相同也是N，每个卷积核的尺寸为C∗K∗K，N个卷积核的总参数量为N∗C∗K∗K

，输入map与输出map的连接方式如下图左所示，图片来自[链接](https://www.researchgate.net/figure/The-transformations-within-a-layer-in-DenseNets-left-and-CondenseNets-at-training-time_fig2_321325862)：

![Convolution VS Group Convolution](https://s2.ax1x.com/2019/01/08/FLPc1x.png)

**Group Convolution**顾名思义，则是对输入feature map进行分组，然后每组分别卷积。假设输入feature map的尺寸仍为C∗H∗W

，输出feature map的数量为N个，如果设定要分成G个groups，则每组的输入feature map数量为CG，每组的输出feature map数量为NG，每个卷积核的尺寸为CG∗K∗K，卷积核的总数仍为N个，每组的卷积核数量为NG，卷积核只与其同组的输入map进行卷积，卷积核的总参数量为N∗CG∗K∗K，可见，**总参数量减少为原来的** 1G

，其连接方式如上图右所示，group1输出map数为2，有2个卷积核，每个卷积核的channel数为4，与group1的输入map的channel数相同，卷积核只与同组的输入map卷积，而不与其他组的输入map卷积。

# Group Convolution的用途

1. **减少参数量**，分成G

组，则该层的参数量减少为原来的1G

- **Group Convolution可以看成是structured sparse**，每个卷积核的尺寸由C∗K∗K
  变为CG∗K∗K，可以将其余(C−CG)∗K∗K- 的参数视为0，有时甚至可以在减少参数量的同时获得更好的效果（相当于**正则**）。

- 当分组数量等于输入map数量，输出map数量也等于输入map数量，即G=N=C
  、N个卷积核每个尺寸为1∗K∗K- 时，Group Convolution就成了**Depthwise Convolution**，参见[MobileNet](https://arxiv.org/abs/1704.04861)和[Xception](https://arxiv.org/abs/1610.02357)等，**参数量进一步缩减**，如下图所示  
  ![Depthwise Separable Convolution](https://s2.ax1x.com/2019/01/08/FLkxED.png)

- 更进一步，如果分组数G=N=C
  ，同时卷积核的尺寸与输入map的尺寸相同，即K=H=W，则输出map为C∗1∗1即长度为C
1. 的向量，此时称之为**Global Depthwise Convolution（GDC）**，见[MobileFaceNet](https://arxiv.org/abs/1804.07573)，可以看成是**全局加权池化**，与 **Global Average Pooling（GAP）** 的不同之处在于，GDC **给每个位置赋予了可学习的权重**（对于已对齐的图像这很有效，比如人脸，中心位置和边界位置的权重自然应该不同），而GAP每个位置的权重相同，全局取个平均，如下图所示：

![global average pooling](https://s2.ax1x.com/2019/01/08/FLEneK.png)

以上。
