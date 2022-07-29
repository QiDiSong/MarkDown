## login服务器如何拷贝文件至sh-ai01

1. 登录sh-ai01服务器，https://sh-etxsrv01.armchina.com/etx/ 
   
   输入用户和密码后进入CentOS7，打开terminal

2. 使用自己的账户，通过copy_from_colo.py文件，将文件从login服务器中拷贝至sh-ai01服务器上。拷贝过程中需要输入自己的用户密码。
   
   copy_from_colo.py --src_dir /project/ai/emulation/user_name/FILE_TRANSFER --dest_dir /dest_path
   
   > 例如：copy_from_colo.py --src_dir /project/ai/emulation/qidson01/lib --dest_dir /home/qidson01/Desktop/
   
   --src_dir: login服务器上用于拷贝文件的中转路径
   
   --dest_dir: sh-ai01服务器上的目标路径，可以指定自己目录下的任何路径
   
   ![image-20220429163625791](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220429163625791.png)

3. 将拷贝到sh-ai01服务器上的文件，通过scp命令拷贝至开发板  libaipudrv.so
   
   scp /FILES_TRANSFERED_PATH/FILES_TRANSFERED root@10.188.100.220:/home/a011911/qidson01/file_transfer
   
   > 例如：scp /home/qidson01/Desktop/lib/* root@10.188.100.220:/home/a011911/qidson01/file_transfer
   > 
   > 需要输入PC机的密码，密码为**ARMUlike2020**
   
   ![image-20220429170052112](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220429170052112.png)

4. 登录PC机，以**a011911**或者**root**用户身份进入，密码为**ARMUlike2020**

> ssh a011911@10.188.100.220
> 
> ssh root@10.188.100.220
> 
> ARMUlike2020（password）

5. 登录开发板，以root用户身份进入

​        ssh root@10.188.100.223

​        **密码为空，回车即可登录进去**

6. 进入板子目录，并创建属于自己的文件夹，以用户名命名（首次传输文件需要先创建自己的工作文件夹，以免影响他人）
   
   /home/mnt_nfs/a011911
   
   mkdir user_name

7. 进入/home/mnt_nfs路径（Linux PC机的a011911目录挂载在开发板的/home/mnt_nfs/a011911下面）。进入该目录即可查看自己传输的文件
   
   ![image-20220509175227623](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220509175227623.png)

8. 如果在板子上生成的profiler数据，可以拷贝出来，放到szc里面使用aipu_profiler命令生成html文件，首先是从板子上拷贝至Linux服务器中
   
   scp root@10.188.100.220:/home/a011911/qidson01/outofbox/benchmark_SONG_perfData.bin /home/qidson01/Desktop

![image-20220519171904944](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220519171904944.png)

9. 通过copy文件将perf_data.bin拷贝至szc的emulation文件夹下
   
   copy_to_colo.py --src_dir ./benchmark_SONG_perfData.bin --dest_dir /project/ai/emulation/qidson01
   
   ![image-20220519172004546](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220519172004546.png)

10. 将perf_data.bin文件拷贝到自己的工作目录中，然后使用aipu_profiler命令生成html文件，此过程需要对应的graph.json文件
    
    aipu_profiler graph.json benchmark_SONG_perfData.bin 
    
    ![image-20220519172110237](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220519172110237.png)

11. 用firefox命令打开对应的html文件

![image-20220519172145938](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220519172145938.png)