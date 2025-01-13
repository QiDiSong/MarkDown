LD

## 1、什么是ld？它有什么作用？

  ld是GNU binutils工具集中的一个，是众多Linkers（链接器）的一种。完成的功能自然也就是链接器的基本功能：把各种目标文件和库文件链接起来，并重定向它们的数据，完成符号解析。Linking其实主要就是完成四个方面的工作：storage allocation、symbol management、libraries、relocation。

  ld可以识别一种Linker command Language表示的linker scriopt文件来显式的控制链接的过程。通过BFD（Binary Format Description）库，ld可以读取和操作COFF（common object file format）、ELF（executable and linking format）、a.out等各种格式的目标文件。

## 2、常用的选项

> -b TARGET 设置目标文件的文件格式
>
> -e ADDRESS 设置目标文件的开始地址
>
> -EB 链接big-endian的目标文件
>
> -EL 链接small-endian的目标文件
>
> **-l LIBNAME  创建执行程序时要链接的库文件（比如某个库为test，则可以为-ltest）**
>
> **-L DIRECTORY 寻找要链接的库文件时搜索的文件路径**
>
> **-o FILE 设置输出文件的名字**
>
> -s 去除输出文件中的所有符号信息
>
> -S 去除输出文件中的调试符号信息
>
> -T FILE 读取链接描述脚本，以确定符号等的定位地址
>
> **-v 输出ld的版本信息**
>
> -x 去除所有的局部符号信息
>
> -X 去除临时的局部符号信息，默认情况下会设置这个选项
>
> -Bstatic  创建的输出文件链接静态链接库
>
> -Bdynamic 创建的输出文件链接动态链接库
>
> -Tbss ADDRESS 设置section bss的起始地址
>
> -Tdata ADDRESS 设置section data的起始地址
>
> -Ttext ADDRESS 设置section text的起始地址

## 3、链接描述脚本

  链接描述脚本描述了各个输入文件的各个section如何映射到输出文件的各section中，并控制输出文件中section和符号的内存布局。

  目标文件中每个section都有名字和大小，而且可以标识为loadable（表示该section可以加载到内存中）、allocatable（表示必须为这个section开辟一块空间，但是没有实际内容下载到这里）。如果不是loadable或者allocatable，则一般含有调试信息。

  每个有loadable或allocatable标识的输出section有两种地址，一种是VMA（Virtual Memory Address），这种地址是输出文件运行时section的运行地址；一种是LMA（Load Memory Address），这种地址是加载输出文件时section的加载地址。一般，这两种地址相同。但在嵌入式系统中，经常存在执行地址和加载地址不一致的情况。如把输出文件加载到开发板的flash存储器中（地址由LMA指定），但运行时，要把flash存储器中的输出文件复制到SDRAM中运行（地址有VMA指定）。

  在链接脚本中使用注释，可以用“/*...*/”。

  每个目标文件有许多符号，每个符号有一个名字和一个地址，一个符号可以是定义的，也可以是未定义的。对于普通符号，需要一个特殊的标识，因为在目标文件中，普通符号没有一个特定的输入section。链接器会把普通符号处理成好像它们都在一个叫做COMMON的section中。

## 3、链接描述脚本

LIBRARY_PATH和LD_LIBRARY_PATH环境变量的区别

LIBRARY_PATH和LD_LIBRARY_PATH是Linux下的两个环境变量，二者的含义和作用分别如下：

LIBRARY_PATH环境变量用于在***程序编译期间\***查找动态链接库时指定查找共享库的路径，例如，指定gcc编译需要用到的动态链接库的目录

```
export LIBRARY_PATH=LIBDIR1:LIBDIR2:$LIBRARY_PATH
```

LD_LIBRARY_PATH环境变量用于在***程序加载运行期间\***查找动态链接库时指定除了系统默认路径之外的其他路径，注意，LD_LIBRARY_PATH中指定的路径会在系统默认路径之前进行查找。设置方法如下（其中，LIBDIR1和LIBDIR2为两个库目录）

```
export LD_LIBRARY_PATH=LIBDIR1:LIBDIR2:$LD_LIBRARY_PATH
```

开发时，设置LIBRARY_PATH，以便gcc能够找到编译时需要的动态链接库。

发布时，设置LD_LIBRARY_PATH，以便程序加载运行时能够自动找到需要的动态链接库。