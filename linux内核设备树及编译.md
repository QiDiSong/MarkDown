**1、设备树的概念**

​    在内核源码中，存在大量对板级细节信息描述的代码。这些代码充斥在/arch/arm/plat-xxx和/arch/arm/mach-xxx目录，对内核而言这些platform设备、resource、i2c_board_info、spi_board_info以及各种硬件的platform_data绝大多数纯属垃圾冗余代码。为了解决这一问题，ARM内核版本3.x之后引入了原先在Power PC等其他体系架构已经使用的Flattened Device Tree。

​    开源文档中对设备树的描述是，一种描述硬件资源的数据结构，它通过bootloader将硬件资源传给内核，使得内核和硬件资源描述相对独立。

​     Device Tree可以描述的信息包括CPU的数量和类别、内存基地址和大小、总线和桥、外设连接、中断控制器和中断使用情况、GPIO控制器和GPIO使用情况、Clock控制器和Clock使用情况。

​     另外，设备树对于可热插拔的设备不进行具体描述，它只描述用于控制该热插拔设备的控制器。

​     设备树的主要优势：对于同一SOC的不同主板，只需更换设备树文件.dtb即可实现不同主板的无差异支持，而无需更换内核文件。

*（注：要使得3.x之后的内核支持使用设备树，除了内核编译时需要打开相对应的选项外，bootloader也需要支持将设备树的数据结构传给内核。）*

**2、设备树的组成和使用**



​    设备树包含DTC（device tree compiler），DTS（device tree source和DTB（device tree blob）。其对应关系如下图所示：

​                                ![img](https://leanote.com/api/file/getImage?fileId=57f737e8ab644106a000aced)

## 2.1 DTS和DTSI(源文件)

​    .dts文件是一种ASCII文本对Device Tree的描述，放置在内核的/arch/arm/boot/dts目录。一般而言，一个.dts文件对应一个ARM的machine。

​     由于一个SOC可能有多个不同的电路板（ .dts文件为板级定义， .dtsi文件为SoC级定义），而每个电路板拥有一个 .dts。这些dts势必会存在许多共同部分，为了减少代码的冗余，设备树将这些共同部分提炼保存在.dtsi文件中，供不同的dts共同使用。.dtsi的使用方法，类似于C语言的头文件，在dts文件中需要进行include .dtsi文件。当然，dtsi本身也支持include 另一个dtsi文件。

## 2.2 DTC (编译工具)

​    DTC为编译工具，dtc编译器可以把dts文件编译成为dtb，也可把dtb编译成为dts文件。在3.x内核版本中，DTC的源码位于内核的scripts/dtc目录，内核选中CONFIG_OF，编译内核的时候，主机可执行程序DTC就会被编译出来。 即scripts/dtc/Makefile中

1. hostprogs-y := dtc
2. always := $(hostprogs-y) 

​    在内核的arch/arm/boot/dts/Makefile中，若选中某种SOC，则与其对应相关的所有dtb文件都将编译出来。在linux下，make dtbs可单独编译dtb。以下截取了TEGRA平台的一部分。

1. ifeq ($(CONFIG_OF),y)
2. dtb-$(CONFIG_ARCH_TEGRA) += tegra20-harmony.dtb \
3. tegra30-beaver.dtb \
4. tegra114-dalmore.dtb \
5. tegra124-ardbeg.dtb 

​    在2.6.x版本内核中，只在powerpc架构下使用了设备树，DTC的源码位于内核的arch/powerpc/boot/dtc-src目录，编译内核后，可将DTC编译出来，DTC编译工具位于arch/powerpc/boot目录下。

## 2.3 DTB (二进制文件)

​    DTC编译.dts生成的二进制文件（.dtb），bootloader在引导内核时，会预先读取.dtb到内存，进而由内核解析。

​    在2.6.x版本内核中，在powerpc架构下，dtb文件可以单独进行编译，编译命令格式如下：



dtc [-I input-format] [-O output-format][-o output-filename] [-V output_version] input_filename

参数说明

input-format：

\- “dtb”: “blob” format

\- “dts”: “source” format.

\- “fs” format.

output-format：

\- “dtb”: “blob” format

\- “dts”: “source” format

\- “asm”: assembly language file

output_version：

定义”blob”的版本，在dtb文件的字段中有表示，支持1　2　3和16,默认是3,在16版本上有许多特性改变

(1) Dts编译生成dtb

./dtc -I dts -O dtb -o B_dtb.dtb A_dts.dts

把A_dts.dts编译生成B_dtb.dtb

(2) Dtb编译生成dts

./dtc -I dtb -O dts -o A_dts.dts A_dtb.dtb

把A_dtb.dtb反编译生成为A_dts.dts

​    在linux 3.x内核中，可以使用make的方式进行编译。

## 2.4 Bootloader(boottloader支持)

  Bootloader需要将设备树在内存中的地址传给内核。在ARM中通过bootm或bootz命令来进行传递。  

  bootm [kernel_addr] [initrd_address] [dtb_address]，其中kernel_addr为内核镜像的地址，initrd为initrd的地址，dtb_address为dtb所在的地址。若initrd_address为空，则用“-”来代替。



**3、linux内核对硬件的描述方式**

​    在以前的内核版本中：
1）内核包含了对硬件的全部描述；
2）bootloader会加载一个二进制的内核镜像，并执行它，比如uImage或者zImage；
3）bootloader会提供一些额外的信息，成为ATAGS，它的地址会通过r2寄存器传给内核；
  ATAGS包含了内存大小和地址，kernel command line等等；
4）bootloader会告诉内核加载哪一款board，通过r1寄存器存放的machine type integer；
5）U-Boot的内核启动命令：bootm <kernel img addr>
6）Barebox变量：bootm.image (?)
[![img](http://upload.semidata.info/www.eefocus.com/blog/media/201410/331381.jpg)](http://upload.semidata.info/www.eefocus.com/blog/media/201410/331381.jpg)

现今的内核版本使用了Device Tree：
1）内核不再包含对硬件的描述，它以二进制的形式单独存储在另外的位置：the device tree blob
2）bootloader需要加载两个二进制文件：内核镜像和DTB
  内核镜像仍然是uImage或者zImage；
  DTB文件在arch/arm/boot/dts中，每一个board对应一个dts文件；
3）bootloader通过r2寄存器来传递DTB地址，通过修改DTB可以修改内存信息，kernel command line，以及潜在的其它信息；
4）不再有machine type；
5）U-Boot的内核启动命令：bootm <kernel img addr> - <dtb addr>
6）Barebox变量：bootm.image,bootm.oftree
[![img](http://upload.semidata.info/www.eefocus.com/blog/media/201410/331382.jpg)](http://upload.semidata.info/www.eefocus.com/blog/media/201410/331382.jpg)

​    有些bootloader不支持Device Tree，或者有些专门给特定设备写的版本太老了，也不包含。为了解决这个问题，CONFIG_ARM_APPENDED_DTB被引进。 
  它告诉内核，在紧跟着内核的地址里查找DTB文件；
  由于没有built-in Makefile rule来产生这样的内核，因此需要手动操作：
​    cat arch/arm/boot/zImage arch/arm/boot/dts/myboard.dtb > my-zImage
​    mkimage ... -d my-zImage my-uImage
  (cat这个命令，还能够直接合并两个mp3文件哦！so easy！)
另外，CONFIG_ARM_ATAG_DTB_COMPAT选项告诉内核去bootloader里面读取ATAGS，并使用它们升级DT。



**4、DTB加载及解析过程**



![img](https://leanote.com/api/file/getImage?fileId=57f737e8ab644106a000acf2)

  先从uboot里的do_bootm出发，根据之前描述，DTB在内存中的地址通过bootm命令进行传递。在bootm中，它会根据所传进来的DTB地址，对DTB所在内存做一系列操作，为内核解析DTB提供保证。上图为对应的函数调用关系图。

  在do_bootm中，主要调用函数为do_bootm_states，第四个参数为bootm所要处理的阶段和状态。 

  在do_bootm_states中，bootm_start会对lmb进行初始化操作，lmb所管理的物理内存块有三种方式获取。起始地址，优先级从上往下：

1.  环境变量“bootm_low”
2.  宏CONFIG_SYS_SDRAM_BASE（在tegra124中为0x80000000）
3.  gd->bd->bi_dram[0].start

大小：

1.  环境变量“bootm_size”
2.  gd->bd->bi_dram[0].size

  经过初始化之后，这块内存就归lmb所管辖。接着，调用bootm_find_os进行kernel镜像的相关操作，这里不具体阐述。

  还记得之前讲过bootm的三个参数么，第一个参数内核地址已经被bootm_find_os处理，而接下来的两个参数会在bootm_find_other中执行操作。

  首先，bootm_find_other根据第二个参数找到ramdisk的地址，得到ramdisk的镜像；然后根据第三个参数得到DTB镜像，同检查kernel和ramdisk镜像一样，检查DTB镜像也会进行一系列的校验工作，如果校验错误，将无法正常启动内核。另外，uboot在确认DTB镜像无误之后，会将该地址保存在环境变量“fdtaddr”中。

  接着，uboot会把DTB镜像reload一次，使得DTB镜像所在的物理内存归lmb所管理：  

- ①boot_fdt_add_mem_rsv_regions会将原先的内存DTB镜像所在的内存置为reserve，保证该段内存不会被其他非法使用，保证接下来的reload数据是正确的；
- ②boot_relocate_fdt会在bootmap区域中申请一块未被使用的内存，接着将DTB镜像内容复制到这块区域（即归lmb所管理的区域）

> 注：若环境变量中，指定“fdt_high”参数，则会根据该值，调用lmb_alloc_base函数来分配DTB镜像reload的地址空间。若分配失败，则会停止bootm操作。因而，不建议设置fdt_high参数。

  接下来，do_bootm会根据内核的类型调用对应的启动函数。与linux对应的是do_bootm_linux。

- ① boot_prep_linux

​    为启动后的kernel准备参数

- ② boot_jump_linux

![img](https://leanote.com/api/file/getImage?fileId=57f737e8ab644106a000acf7)

  以上是boot_jump_linux的片段代码，可以看出：若使用DTB，则原先用来存储ATAG的寄存器R2，将会用来存储.dtb镜像地址。

  boot_jump_linux最后将调用kernel_entry，将.dtb镜像地址传给内核。

 

  下面我们来看下内核的处理部分：

  在arch/arm/kernel/head.S中，有这样一段：

![img](https://leanote.com/api/file/getImage?fileId=57f737e8ab644106a000acea)

  _vet_atags定义在/arch/arm/kernel/head-common.S中，它主要对DTB镜像做了一个简单的校验。

![img](https://leanote.com/api/file/getImage?fileId=57f737e8ab644106a000acef)

  真正解析处理dbt的开始部分，是setup_arch->setup_machine_fdt。这部分的处理在第五部分的machine_mdesc中有提及。

![img](https://images2015.cnblogs.com/blog/334341/201702/334341-20170215165840816-1184117050.png)

 

  如图，是setup_machine_fdt中的解析过程。

-   解析chosen节点将对boot_command_line进行初始化。
-   解析根节点的{size，address}将对dt_root_size_cells，dt_root_addr_cells进行初始化。为之后解析memory等其他节点提供依据。
-   解析memory节点，将会把节点中描述的内存，加入memory的bank。为之后的内存初始化提供条件。

 

-   解析设备树在函数unflatten_device_tree中完成，它将.dtb解析成device_node结构（第五部分有其定义），并构成单项链表，以供OF的API接口使用。