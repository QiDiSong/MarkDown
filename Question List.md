Question List

1. git pull和git clone都不能使用，且没有信息提示就结束了，而git status git log等命令都可以使用。

   解决方法是自己未登录C14204等远端服务器，对方无法认证自己身份，故不能下载。而自己本地仓库是可以使用git log等不涉及远端服务器的命令的

2. 如何免密push和pull

   git config credential.helper store

3.  离线安装VSCode插件

   在windows下载想要的插件之后，可以用everything搜索/.vscode\extension，找到对应的插件后，拷贝至linux下面的~/.vscode/extension

   另外：需要在插件文件夹下面找到package.json文件，搜索vscode的最小支持版本，修改该支持版本是其支持linux下面安装的vscode版本
   
4. which python & which python3

   > qidson01@c20106 /project/ai/scratch01/qidson01/code/AIPU_runtime_design >which python
   > **/bin/python**
   > qidson01@c20106 /project/ai/scratch01/qidson01/code/AIPU_runtime_design >which python3
   > **/arm/tools/python/python/3.8.5/rhe7-x86_64/bin/python3**

which python查看的是python2的版本

which python3查看的是python3的版本

5. 当pip install GBuilder的时候，

> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >which python3
> /arm/tools/python/python/3.8.5/rhe7-x86_64/bin/python3
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >cd ~/.local/
> bin/   lib/   share/ 
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >cd ~/.local/bin/
> aipu_profiler*          aipugb*                 aipuopt*                aipuspt*                convert-caffe2-to-onnx* unrar@
> aipubinutils*           aipugsim*               aipuparse*              aipusptn*               convert-onnx-to-caffe2* 
> aipubuild*              aipugt*                 aipurun*                codec@                  rar@                    
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >cd ~/.local/lib/
> python2.7/ python3.6/ 
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >cd ~/.local/lib/python
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >module load swdev python/python/3.6.5
> qidson01@c20108 /project/ai/scratch01/qidson01/debug/customer/shuffle_v2 >pip uninstall aipubuild

6. [E] [Parser]: Meets invalid model file or invalid output directory!

   无效的模型文件或者无效的输出目录，可能是cfg文件里面的output_dir文件自己没有写权限，因此上报无效的输出目录

7. 平常用的login的ETX服务器都在深圳那边，网址为：

   https://szc-etxsrv01.armchina.com/etx/

   而sh-ai01和paladin服务器在上海这边，网址为：

   https://sh-etxsrv01.armchina.com/etx/

   这些都是HPC服务器

8. make menuconfig的时候，出现display is too small的问题，将terminal变大即可。（或者不再去用vscode下面的terminal）

   > qidson01@c20108 /project/ai/scratch01/qidson01/platform/from_hobart/linux-xlnx-armchina >make menuconfig
   > scripts/kconfig/mconf  Kconfig
   > Your display is too small to run Menuconfig!
   > It must be at least 19 lines by 80 columns.
   > make[1]: *** [menuconfig] Error 1
   > make: *** [menuconfig] Error 2

9. 当terminal下出现未找到工具链c++-14等信息时，是因为自己的环境中未定义该工具链。需要source一下自己的cshrc文件或者其他 的设置环境的文件。或者直接指定工具链路径。

10. **Linux内核存放位置**
    qidson01@c20108 /project/ai/scratch01/AIPU_BSP/kernel >ls
    linux-4.14.tgz  linux-4.9.168  linux-5.11.18  linux-5.11.18.tgz  linux-xlnx-armchina  tmp
    
11. 如何理解Jenkins

    jenkins是我们公司的一个自动化测试中转服务器，通过cluster把指令发到jenkins服务器，然后再转发到板子上执行，板子的结果返回jenkins，最后上传报告给到user

12. ModuleNotFoundError: No module named 'Test_logger'。找不到Test_logger模块

​		更新pycharm的python3环境：从python3.6.5更新至python3.8.5，之后再次运行pytest的时候，发现报错，找不到Test_logger模块。

​		此模块为arm公司针对自己的产品，对原来python的log信息进行包装之后，重新发布的一个模块。而更新完python环境后，新的python环境包里面没有这个Test_logger模块，因此，需要自己安装模块。

​		首先定位到Test_logger模块的文件处，发现位于AIPU_common/verification/python/Test_Logger地方。然后在此文件位置，发现有一个setup.py文件。此文件用于安装Test_logger模块到python环境处。

> ​		命令： python3 setup.py install --user

​		安装完此模块之后，就可以像numpy一样，不用写路径就可以直接import此模块了。

> 例如： from Test_Logger import logger as log

13. 设计自定义的python模块文件示例，需要填以下内容：

    > from setuptools import setup, find_packages
    >
    > setup(
    >     name='Test_Logger',
    >     version='1.3',
    >     description='Test logger',
    >     install_requires=[],
    >     packages=find_packages(),
    >     package_data={"Test_Logger": ["resources/*"]},
    >     entry_points={'pytest11': ['Test_Logger = Test_Logger.hooks']},
    >     include_package_data=True,
    > )

14. 安装show

    python3 setup.py install --user

    安装完show之后，需要source一下环境变量才可以使用，update一下当前的环境变量

15. aipuparse --cfg model_name.cfg

    例如：aipuparse --cfg channel_shuffle_run.cfg

    这个model_name需要写parse的部分，这样可以用aipuparse命令只跑parser的部分，解析出float IR

    float IR 里面可以查看这个IR里面的输入输出以及　layer_bottom, layer_top等信息。

    cfg文件示例如下：

    > [Common]
    > model_type = onnx
    > model_name = ChannelShuffle
    > detection_postprocess = 
    > model_domain = image_classification
    > input_model = ./ChannelShuffle_1.onnx
    > output_dir = ./parser/

​		解析得到的float IR如下：

model_name=ChannelShuffle
model_domain=image_classification
layer_number=5
precision=float32
model_bin=./ChannelShuffle.bin
input_tensors=[Placeholder]
output_tensors=[split_0_port_0_post_trans]

layer_id=0
layer_name=Placeholder
layer_type=Inputi[]
layer_bottom=[]
layer_bottom_shape=[]
layer_bottom_type=[]
layer_top=[Placeholder]
layer_top_shape=[[2,12,224,200]]
layer_top_type=[float32]

layer_id=1
layer_name=Placeholder_post_transpose
layer_type=Transpose
layer_bottom=[Placeholder]
layer_bottom_shape=[[2,12,224,200]]
layer_bottom_type=[float32]
layer_top=[Placeholder_post_transpose]
layer_top_shape=[[2,224,200,12]]
layer_top_type=[float32]
perm=[0,2,3,1]

layer_id=2
layer_name=Reshape_1
layer_type=ChannelShuffle
layer_bottom=[Placeholder_post_transpose]
layer_bottom_shape=[[2,224,200,12]]
layer_bottom_type=[float32]
layer_top=[Reshape_1_0]
layer_top_shape=[[2,224,200,12]]
layer_top_type=[float32]
group=3
splits=1

layer_id=3
layer_name=split_0
layer_type=Split
layer_bottom=[Reshape_1_0]
layer_bottom_shape=[[2,224,200,12]]
layer_bottom_type=[float32]
layer_top=[split_0_0]
layer_top_shape=[[2,224,200,12]]
layer_top_type=[float32]
axis=3
splits=[12]

layer_id=4
layer_name=split_0_port_0_post_trans
layer_type=Transpose
layer_bottom=[split_0_0]
layer_bottom_shape=[[2,224,200,12]]
layer_bottom_type=[float32]
layer_top=[split_0_port_0_post_trans]
layer_top_shape=[[2,12,224,200]]
layer_top_type=[float32]
perm=[0,3,1,2]



input_model = /project/ai/scratch01/Model/kws_gru/onnx/1_6/model/kws_gru.onnx
input = fingerprint_input:0
input_shape = [1, 49, 13]
output = Softmax:0

input: {'name': ['fingerprint_input'], 'shape': [[1, 49, 13]], 'dtype': ['tensor(float)']}
output: {'shape': [[1, 12]], 'dtype': ['tensor(float)']}

input = Placeholder:0
input_shape = [1, 49, 13]

input: {'name': ['Placeholder'], 'shape': [[2, 12, 224, 200]], 'dtype': ['tensor(float)']}
output: {'shape': [[2, 12, 224, 200]], 'dtype': ['tensor(float)']}

model_bin=./ChannelShuffle.bin
input_tensors=[Placeholder]
output_tensors=[split_0_port_0_post_trans]

16. 得到temp.cfg之后，可以直接调用aipu_simulator执行cfg文件

    等价于aipurun -i input.bin -o output.bin

    命令如下：

> /project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_z2 temp.cfg

17.  aipurun的时候，未指定simulator导致run simulator失败，当指定好simulator之后即可： --simulator /project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_z2

    ![image-20220609111516392](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220609111516392.png)

18. 
