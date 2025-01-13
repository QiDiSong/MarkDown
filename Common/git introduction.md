环境变量的作用：可以在任意地方都可以使用此环境变量

Git是目前最先进的分布式管理代码工具

分布式代码管理：没有服务器，每一个人都有此代码仓的所有代码提交记录，防止服务器一旦挂掉就没法提交代码。

Git本地有三个工作区域：工作目录（Working Directory）、暂存区（Stage/Index）**资源库（Repository或者Git Directory）**如果再加上远程的git仓库（Remote Directory），那就是4个工作区，四个区域的转换关系如下：

![image-20220227184339396](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220227184339396.png)

Workspace：平时写代码的地方，存代码的地方

Index/Stage：暂存区，用于临时存放代码的改动，事实上它只是一个文件，保存即将提交到文件列表信息

![image-20220227190400665](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220227190400665.png)

 ![image-20220227190536650](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220227190536650.png)

![image-20220227191750381](C:\Users\33010\AppData\Roaming\Typora\typora-user-images\image-20220227191750381.png)

设置本机绑定SSH公钥，可以实现免密码登录

否则每次push都需要输入用户名和密码