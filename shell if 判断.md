## shell if 判断

**注意事项**

> 1、[ ]表示条件测试。注意这里的空格很重要。要注意在'['后面和']'前面都必须要有空格
> 2、在shell中，then和fi是分开的语句。如果要在同一行里面输入，则需要用分号将他们隔开。
> 3、注意if判断中对于变量的处理，需要加引号，以免一些不必要的错误。没有加双引号会在一些含空格等的字符串变量判断的时候产生错误。比如[ -n "$var" ]如果var为空会出错
> 4、判断是不支持浮点值的
> 5、如果只单独使用>或者<号，系统会认为是输出或者输入重定向，虽然结果显示正确，但是其实是错误的，因此要对这些符号进行转意
> 6、在默认中，运行if语句中的命令所产生的错误信息仍然出现在脚本的输出结果中
> 7、使用-z或者-n来检查长度的时候，没有定义的变量也为0
> 8、空变量和没有初始化的变量可能会对shell脚本测试产生灾难性的影响，因此在不确定变量的内容的时候，在测试号前使用-n或者-z测试一下
> 9、? 变量包含了之前执行命令的退出状态（最近完成的前台进程）（可以用于检测退出状态）

- 文件/目录判断

  ```bash
  # 常用的：
  [ -a FILE ] 如果 FILE 存在则为真。
  [ -d FILE ] 如果 FILE 存在且是一个目录则返回为真。
  [ -e FILE ] 如果 指定的文件或目录存在时返回为真。
  [ -f FILE ] 如果 FILE 存在且是一个普通文件则返回为真。
  [ -r FILE ] 如果 FILE 存在且是可读的则返回为真。
  [ -w FILE ] 如果 FILE 存在且是可写的则返回为真。（一个目录为了它的内容被访问必然是可执行的）
  [ -x FILE ] 如果 FILE 存在且是可执行的则返回为真。
  
  # 不常用的：
  [ -b FILE ] 如果 FILE 存在且是一个块文件则返回为真。
  [ -c FILE ] 如果 FILE 存在且是一个字符文件则返回为真。
  [ -g FILE ] 如果 FILE 存在且设置了SGID则返回为真。
  [ -h FILE ] 如果 FILE 存在且是一个符号符号链接文件则返回为真。（该选项在一些老系统上无效）
  [ -k FILE ] 如果 FILE 存在且已经设置了冒险位则返回为真。
  [ -p FILE ] 如果 FILE 存并且是命令管道时返回为真。
  [ -s FILE ] 如果 FILE 存在且大小非0时为真则返回为真。
  [ -u FILE ] 如果 FILE 存在且设置了SUID位时返回为真。
  [ -O FILE ] 如果 FILE 存在且属有效用户ID则返回为真。
  [ -G FILE ] 如果 FILE 存在且默认组为当前组则返回为真。（只检查系统默认组）
  [ -L FILE ] 如果 FILE 存在且是一个符号连接则返回为真。
  [ -N FILE ] 如果 FILE 存在 and has been mod如果ied since it was last read则返回为真。
  [ -S FILE ] 如果 FILE 存在且是一个套接字则返回为真。
  [ FILE1 -nt FILE2 ] 如果 FILE1 比 FILE2 新, 或者 FILE1 存在但是 FILE2 不存在则返回为真。
  [ FILE1 -ot FILE2 ] 如果 FILE1 比 FILE2 老, 或者 FILE2 存在但是 FILE1 不存在则返回为真。
  [ FILE1 -ef FILE2 ] 如果 FILE1 和 FILE2 指向相同的设备和节点号则返回为真。
  ```

- 字符串判断

  ```bash
  [ -z STRING ] 如果STRING的长度为零则返回为真，即空是真
  [ -n STRING ] 如果STRING的长度非零则返回为真，即非空是真
  [ STRING1 ]　 如果字符串不为空则返回为真,与-n类似
  [ STRING1 == STRING2 ] 如果两个字符串相同则返回为真
  [ STRING1 != STRING2 ] 如果字符串不相同则返回为真
  [ STRING1 < STRING2 ] 如果 “STRING1”字典排序在“STRING2”前面则返回为真。
  [ STRING1 > STRING2 ] 如果 “STRING1”字典排序在“STRING2”后面则返回为真。
  ```

- 数值判断

  ```bash
  [ INT1 -eq INT2 ] INT1和INT2两数相等返回为真 ,=
  [ INT1 -ne INT2 ] INT1和INT2两数不等返回为真 ,<>
  [ INT1 -gt INT2 ] INT1大于INT2返回为真 ,>
  [ INT1 -ge INT2 ] INT1大于等于INT2返回为真,>=
  [ INT1 -lt INT2 ] INT1小于INT2返回为真 ,<
  [ INT1 -le INT2 ] INT1小于等于INT2返回为真,<=
  ```

- 逻辑判断

  ```bash
  [ ! EXPR ] 逻辑非，如果 EXPR 是false则返回为真。
  [ EXPR1 -a EXPR2 ] 逻辑与，如果 EXPR1 and EXPR2 全真则返回为真。
  [ EXPR1 -o EXPR2 ] 逻辑或，如果 EXPR1 或者 EXPR2 为真则返回为真。
  [ ] || [ ] 用OR来合并两个条件
  [ ] && [ ] 用AND来合并两个条件
  ```

- 其他判断

  ```bash
  [ -t FD ] 如果文件描述符 FD （默认值为1）打开且指向一个终端则返回为真
  [ -o optionname ] 如果shell选项optionname开启则返回为真
  ```