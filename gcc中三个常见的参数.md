在本文中， 我们来聊聊gcc中三个常见的参数， 也即-I（大写的i）, -L（大写的l）和-l（小写的l） 



​    **一. 先说 -I  (注意是大写的i)**

​    我们先来看简单的程序：

​    main.c:

```cpp
#include <stdio.h>  



#include "add.h"  



  



int main()  



{  



    int a = 1;  



    int b = 2;  



    int c = add(a, b);  



  



    printf("sum is %d\n", c);  



  



    return 0;  



}  
```



add.c：



```cpp
int add(int x, int y)  



{  



    return x + y;  



}  
```


   add.h:





```html
int add(int x, int y);
```


 编译链接运行如下：





```cpp
[taoge@localhost test]$ pwd  



/home/taoge/test  



[taoge@localhost test]$ ls  



add.c  add.h  main.c  



[taoge@localhost test]$ gcc main.c add.c  



[taoge@localhost test]$ ./a.out   



sum is 3  



[taoge@localhost test]$   
```


我们看到， 一切正常。 gcc会在程序当前目录、/usr/include和/usr/local/include目录下查找add.h文件， 刚好有， 所以ok.





我们进行如下操作后再编译， 却发现有误， 不怕， 我们用-I就行了：



```cpp
[taoge@localhost test]$ ls  



add.c  add.h  a.out  main.c  



[taoge@localhost test]$ rm a.out; mkdir inc; mv add.h inc  



[taoge@localhost test]$ ls  



add.c  inc  main.c  



[taoge@localhost test]$ gcc main.c add.c  



main.c:2:17: error: add.h: No such file or directory  



[taoge@localhost test]$   



[taoge@localhost test]$   



[taoge@localhost test]$   



[taoge@localhost test]$ gcc -I ./inc/ main.c add.c   



[taoge@localhost test]$ ls  



add.c  a.out  inc  main.c  



[taoge@localhost test]$ ./a.out   



sum is 3  



[taoge@localhost test]$   
```


上面把add.h移动到inc目录下后， gcc就找不到add.h了， 所以报错。 此时，要利用-I来显式指定头文件的所在地，  -I就是用来干这个的：告诉gcc去哪里找头文件。





二. 再来说-L(注意是大写的L)

​    我们上面已经说了， -I是用来告诉gcc去哪里找头文件的， 那么-L实际上也很类似， 它是用来告诉gcc去哪里找库文件。 通常来讲， gcc默认会在程序当前目录、/lib、/usr/lib和/usr/local/lib下找对应的库。 -L的意思很明确了， 就不在赘述了。

​    三. 最后说说-l （注意是小写的L）


    我们之前讨论过Linux中的静态库和动态库， -l的作用就是用来指定具体的静态库、动态库是哪个。 