Windows中我们常用vs来编译编写好的C和C++代码；vs把编辑器，[编译器](https://so.csdn.net/so/search?q=编译器&spm=1001.2101.3001.7020)和调试器等工具都集成在这一款工具中，在Linux下我们能用什么工具来编译所编写好的代码呢，其实Linux下这样的工具有很多，但我们只介绍两款常用的工具，它们分别是gcc和g++.

## 工具用法介绍

[gcc](https://so.csdn.net/so/search?q=gcc&spm=1001.2101.3001.7020)和g++的用法都是一样的，在这里我们只介绍gcc：
![这里写图片描述](https://img-blog.csdn.net/20170101110932265?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上图介绍了相关指令和参数以及该条指令所做的对应的事情。

1. gcc -E filename.c -o filename.i
   将c文件转化成C++文件,这个过程也叫做预处理过程
2. gcc -S filename.i -o filename.s
   将预处理过程生成的.i后缀的文件转化成汇编文件，里面存储的是相应的汇编代码，这个过程叫做编译。
3. gcc -c filename.s -o filename.o
   将汇编文件中的汇编代码翻译成相应的机器语言，这个过程叫做汇编。
4. gcc filename.o -o filename.exe
   这条指令是完成链接这个过程的，它通过链接器ld将运行程序的目标文件和库文件链接在一起，生成最后的可执行文件
5. 生成可执行文件后，我们就能够调用相应的程序了。
   **注意：由于g++和gcc的用法相同，所以在这里我们就不直接介绍了**

## gcc和g++的区别

### 编译c程序

熟悉C++的人应该都知道，C++是C语言的超集，编写C/C++代码的时候，有人用gcc，也有人用g++,我们先来看看gcc和g++是否都能编译C++和C代码：
![这里写图片描述](https://img-blog.csdn.net/20170101112622358?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
上图中，我们发现用gcc和g++分别编译test.c这个c文件，发现都是可执行的，实际上对于C文件gcc和g++所做的事情确实是一样的，g++在编译C文件时调用了gcc.

### 编译cpp程序

既然二者对c程序来说都一样的，那为什么两者都存在而不是只存在一个呢,不要着急，下面我们来看看他们分别是如何来编译C++程序的.
首先我们直接编译c文件生成可执行程序：
![这里写图片描述](https://img-blog.csdn.net/20170101113734015?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
我们发现gcc报错，而g++没有报错，并且可以执行。

下面我们来看看它们的具体步骤以及错误原因:

1. 预处理
   ![这里写图片描述](https://img-blog.csdn.net/20170101114705808?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
   在预处理阶段两条命令都能够成功，并且我们观察gcc和g++各自产生的.i后缀的文件，它们的内容都是相同的，所以我们能够发现gcc和g++在cpp程序中它们做了相同的事情。
2. 编译
   ![这里写图片描述](https://img-blog.csdn.net/20170101115507650?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
   我们发现gcc编译test1.i(.cpp生成)的会出现大篇幅的错误，图片中的错误主要是说无法找到cout函数的库文件，而g++去可以通过编译并且生成汇编文件,这件事情说明gcc无法自动和c++的库文件进行连接，导致了库函数没有申明的错误.
3. 汇编
   这个过程应该都没有问题，因为这个过程只是将后缀为.s文件中的汇编语言转换成了相应的机器语言。所以gcc和g++应该在这个过程中做了同样的事情。
4. 链接
   ![这里写图片描述](https://img-blog.csdn.net/20170101120255801?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYml0X2NsZWFyb2Zm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
   这个阶段也出现了问题，用gcc将.cpp后缀产生的.o文件转换成可执行文件时出现了错误，而g++却可以转换成功并且能够正确执行。这个还是因为gcc无法将库文件与.o后缀的文件关联在一起生成可执行程序，而g++可以完成这项工作。

## 总结

gcc和g++的区别主要是在对cpp文件的编译和链接过程中，因为cpp和c文件中库文件的命名方式不同，那为什么g++既可以编译C又可以编译C++呢，这时因为g++在内部做了处理，默认编译C++程序，但如果遇到C程序，它会直接调用gcc去编译.