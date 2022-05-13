### **开篇**

本文引用的内核代码参考来自版本 linux-5.15.4 。

在用户空间，用指令 insmod 来向内核空间安装一个内核模块，其使用方法如下：

```text
insmod xx.ko  /* 向内核空间安装模块 xx */
```

**注意，加载内核模块需要具有 root 权限，否则会加载失败。**

当调用 “insmod xx.ko” 来安装 “xx.ko” 内核模块时，insmod 会首先利用文件系统的接口，将模块文件的数据读取到用户空间的一段内存中，然后通过系统调用 `sys_init_module` 让内核去处理模块加载的整个过程。

### **系统调用 sys_init_module**

sys_init_module() 函数的原型为：

```text
long sys_init_module(void __user *umod, unsigned long len, const char __user *uargs);
```

参数 umod，是指向用户空间内核模块文件映像数据的内存地址。参数 len，是该文件的数据大小。第三个参数 uargs，是传给模块的参数在用户空间下的内存地址。

函数的具体代码如下（已经将函数名称替换为实际展开后的形式）：

```text
/* <kernel/module.c> */

long sys_init_module(void __user *umod, unsigned long len, const char __user *uargs);
{
  int err;
  struct load_info info = { };

  err = may_init_module();  /* 判断是否有加载模块的权限 */
  if (err)
    return err;

  pr_debug("init_module: umod=%p, len=%lu, uargs=%p\n", umod, len, uargs);

  /* 将模块文件数据复制到内核空间 */
  err = copy_module_from_user(umod, len, &info);
  if (err)
    return err;

  /* 加载模块 */
  return load_module(&info, uargs, 0);
}
```

由以上代码可知，加载模块的工作主要是通过 `load_module` 函数完成的。

该函数完成模块加载的全部任务，原型为：

```text
static int load_module(struct load_info *info, const char __user *uargs, int flags)
```

参数 info 为结构指针，指向存储模块文件数据的结构。参数 uargs，与函数`sys_init_module` 的参数 uargs 相同。参数 flags 为加载标志。

函数的主要功能为：分配模块需要的内存资源，然后将模块加载到内核中。

模块加载成功，返回值为 0。加载失败，则返回错误码（负值）。

### **关键数据结构**

**结构体 struct load_info**

在加载过程中会用到一个类型为 load_info 的结构体变量 info，此变量在模块加载过程中临时记录一些参数。结构体 load_info 的定义如下：

```text
/* <kernel/module-internal.h> */

struct load_info {
  const char *name;
  /* pointer to module in temporary copy, freed at end of load_module() */
  struct module *mod;
  Elf_Ehdr *hdr;     /* 模块文件内容指针 */
  unsigned long len;  /* 模块文件大小（字节数） */
  Elf_Shdr *sechdrs;
  char *secstrings, *strtab;
  unsigned long symoffs, stroffs, init_typeoffs, core_typeoffs;
  struct _ddebug *debug;
  unsigned int num_debug;
  bool sig_ok;
#ifdef CONFIG_KALLSYMS
  unsigned long mod_kallsyms_init_off;
#endif
  struct {
    unsigned int sym, str, mod, vers, info, pcpu;
  } index;
};
```

**结构体 struct module**

此结构体用来管理系统中加载的模块，是一个非常重要的数据结构。

一个 struct module 对象代表着一个内核模块在 Linux 系统的抽象。由于该结构成员变量特别多，只列出了关键的几个成员变量，并做了注释说明，如下：

```text
/* <include/linux/module.h> */

struct module {
  enum module_state state; /* 记录模块加载过程中的不同阶段状态 */
  struct list_head list;   /* 用来将模块链接到系统维护的内核模块链表中 */
  char name[MODULE_NAME_LEN];  /* 模块名称 */
  const struct kernel_symbol *syms;  /* 内核模块导出的符号所在起始地址 */
  const s32 *crcs;  /* 内核模块导出符号的校验码存放地址 */
  struct kernel_param *kp;  /* 内核模块参数所在地址 */
  int (*init)(void);  /* 内核模块初始化函数的指针 */
  struct list_head source_list; /* 用来在内核模块之间建立依赖关系 */
  struct list_head target_list;
  void (*exit)(void);  /* 内核模块退出函数指针 */
};
```

模块加载过程不同阶段的状态，module_state 定义如下：

```text
enum module_state {
  MODULE_STATE_LIVE,  /* 模块成功加载进系统时的状态 */
  MODULE_STATE_COMING,  /* 配置完成，开始加载模块 */
  MODULE_STATE_GOING, /* 加载过程出错，退出加载 */
  MODULE_STATE_UNFORMED,  /* 正在建立加载配置 */
};
```

### **加载函数 load_module**

此函数主要分两部分功能：一部分完成模块加载最核心的任务；第二部分是，模块被加载到系统的后续处理。

**load_module 第一部分**

- **构造模块 ELF 的内存视图**

通过调用 `copy_module_from_user()`函数，将用户空间的模块文件数据复制到内核空间中，从而在内核空间构造出模块的一个 ELF 静态内存视图。也就是 HDR 视图，加载完成后会将其释放。

- **创建字符串表**

字符串表是 ELF 文件中的一个 section，用来保存 ELF 文件中各个 section 的名称或符号。通过调用 `setup_load_info(info, flags);` 会创建这个字符串表，并得到 section 名称字符串表的基地址 secstrings。

- **find_sec 函数**

通过调用此函数，内核寻找某一个 section 在 section header table 中的索引值。分别查找以下 section：“.modinfo”、“__versions”、“.gnu.linkonce.this_module”，保存查到的索引值，以备将来使用。

- **HDR 视图第一次改写**

第一次遍历 section header table 中的所有 entry，修改 entry 中的 sh_addr ，计算语句如下：

```text
shdr->sh_addr = (size_t)info->hdr + shdr->sh_offset;
```

这样每个 entry 中的 sh_addr 指向该 entry 所对应的 section 在 HDR 视图中的实际存储地址。

- **struct module 类型变量 mod 初始化**

结构体 struct module 是一个非常重要的数据结构，内核用来表示一个模块。在 load_module 函数中定义了一个 struct module 类型的变量 mod。调用 `mod = layout_and_allocate(info, flags);` 分配需要的内存，并初始化。

- **HDR 视图的第二次改写**

这次改写中，HDR 视图中绝大多数的 section 会被搬移到新的内存空间中，使得其中 section header table 中各个 entry 的 sh_addr 指向最终的内存地址。

- **模块导出的符号**

模块可以向外部导出自己的符号。如果一个内核向外界导出了自己的符号，那么模块编译工具链负责生成这些导出的符号 section。而这些 section 都带有 SHF_ALLOC 标志，模块在加载过程中会被搬移到 CORE section 区域中。

- **find_symbol 函数**

第一部分，在内核导出的符号表中查找指定的符号。第二部分，在系统已经加载的模块导出的符号表中查找符号。

- **对“未解决的引用”符号的处理**

“未解决的引用符号“，就是模块编译链接生产 .ko 文件时，对于模块中调用的一些函数，链接工具无法在所有的目标文件中找到某个函数的指令码，链接工具会将这个符号标记为”未解决的引用符号“。模块被加载时，内核会解决这些符号。

- **重定位**

主要用来解决静态链接时的符号引用，与动态加载时的实际符号地址不一致的问题。

- **模块参数**

在用 insmod 加载模块时，有时需要向模块传递一些参数，内核模块本身在源代码中必须用宏 module_param 声明模块可以接收的参数。内核加载器可以得到从命令行传过来的实际参数。

- **处理模块间的依赖关系**

内核能跟踪模块间的依赖关系，在模块加载过程中，建立模块之间的依赖关系。

- **模块的版本控制**

版本控制主要用来解决内核模块和内核之间的接口一致性问题。避免模块使用内核已经改变或废弃的接口，而导致加载失败或者存在风险的问题。

- **模块的信息**

模块最终的 ELF 文件中都会有一个名为 ”.modinfo“ 的 section，以文本形式保留着模块的一些信息。加载过程中，内核需要获得 ”.modinfo“ section 中的相关信息，以便进一步处理。包括：**模块的 license**、**模块的 vermagic**。

**load_module 第二部分**

第二部分通过调用 `do_init_module()`函数完成。

在这个函数中，首先调用模块的构造函数 `do_mod_ctors()`。然后调用模块的初始化函数，也就是 mod 中 init 指针指向的函数，初始化函数通过 `do_one_initcall()`完成调用。

模块完成加载之后，HDR 视图和 INIT section 所占的内存空间不再使用，需要释放它们。

HDR 视图在调用 `do_init_module()`之前完成释放（调用 `free_copy(info)`）。INIT section 在 `do_init_module()`结尾完成释放。

模块加载进系统之后，链接到内核维护的模块链表 modules 中，该链表记录着系统中所有已加载的模块。