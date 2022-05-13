aipugb对应四种模式中的build模式：用于生成aipu.bin（二进制文件）

aipugb resnet_50_int8.txt -w resnet_50_int8.bin --target Z3_0901 -D

![b78f992036dcc5a5a8cefb4464bdf28](C:\Users\33010\Desktop\b78f992036dcc5a5a8cefb4464bdf28.png)





跑aipurun的时候，如果不指定target，则默认是Z1_0904,

aipurun的时候，需要指定target版本和simulator版本，两者要保持一致

simulator版本如果不指定的话，则使用本terminal环境的上一次simulator设置的版本。



aiff：一种硬件加速单元，会对每一层的寄存器进行保存，用于加速。