tar命令是类Linux中比较常用的解压与压缩命令。

可以使用命令 (man tar) 命令来进行查看man的基本命令。下面举例说明一下tar 的基本命令。

\#tar -cvf   sysconfig.tar  /etc/sysconfig

命令解释：将目录/etc/sysconfig/目录下的文件打包成文件sysconfig.tar文件，并且放在当前目录中

（可以使用pwd命令查看当前路径，可以使用ls命令来查看当前文件夹）参数解释如下：

-c 创建新的文档。

-v 显示详细的tar处理的文件信息

-f 要操作的文件名


 

\#tar -rvf   sysconfig.tar  /etc/sysconfig/

命令解释：将目录/etc/sysconfig/目录下的文件添加到文件sysconfig.tar文件中去。参数解释如下：

-r 表示增加文件，把要增加的文件追加在压缩文件的末尾。


 

\#tar -tvf sysconfig.tar

命令解释：查看压缩文件sysconfig.tar文件里面的内容参数解释如下：

-t 表示查看文件，查看文件中的文件内容


 

\#tar -xvf sysconfig.tar

命令解释：解压文件sysconfig.tar，将压缩文件sysconfig.tar文件解压到当前文件夹内。参数解释如下：

-x 解压文件。



tar调用程序进行压缩与解压缩。

1、tar调用gzip。

.gz结尾的文件就是调用gzip程序进行压缩的文件，相反文件以.gz结尾的文件需要使用gunzip来进行解压。tar中使用-z参数

来调用gzip程序。在这里通过举例子来进行解释。

\#tar -czvf sysconfig.tar.gz /etc/sysconfig/

命令解释：将目录/etc/sysconfig/打包成一个tar文件包，通过使用-z参数来调用gzip程序，对目录/etc/sysconfig/进行压缩，

压缩成文件sysconfig.tar.gz，并且将压缩成的文件放在当前文件夹内。参数解释如下：

-z 调用gzip程序来压缩文件，压缩后的文件名称以.gz结尾。

\#tar -xzvf sysconfig.tar.gz

命令解释：这条命令是将上一条命令解压。


 


 

2、tar调用bzip2

.bz2结尾的文件就是调用bzip2程序来进行压缩的文件，相反，文件以.bz2结尾的文件需要使用bunzip2来解压。tar中使用-j

参数来调用程序bzip2。

\#tar -cjvf sysconfig.tar.bz2 /etc/sysconfig/

命令解释：将/etc/sysconfig/目录打包成一个tar包，接着使用-j参数调用bzip2来进行压缩文件，对目录/etc/sysconfig/进行

压缩，压缩成文件sysconfig.tar.bz2并将其放在当前目录下。

\#tar -xjvf sysconfig.tar.bz2

命令解释：解压上一个命令生成的压缩包。

![img](https://img-blog.csdn.net/20180125163726526?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfNDAyMzI4NzI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 