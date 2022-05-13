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
