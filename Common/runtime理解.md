runtime理解

**runtime是用来管理和服务你写的代码的代码。他是你写的代码的基础层，由编译器插入你的代码之前，先于你的代码执行，对你的代码的运行环境进行初始化，或者常驻一部分代码以便服务你的代码。**

**总之， runtime 的意思大概就是 「运行期所必需的东西」。**

怎样理解 runtime library 呢？要知道 C、C++ 和 Rust 这类「系统级语言」相比于 JavaScript 这类「应用级语言」最大的特点之一，就在于它们可以胜任嵌入式裸机、操作系统驱动等贴近硬件性质的开发——**而所谓 runtime library，大致就是这时候你没法用的东西**。

回想一下，我们在 C 语言里是怎么写 hello world 的呢？

```c
#include <stdio.h> // 1

int main(void) { // 2
  printf("Hello World!\n"); // 3
}
```

这里面除了最后一个括号，每行都和运行时库有很大关系：

1. `stdio.h` 里的符号是 C 标准库提供的 API，我们可以 include 进来按需使用（但注意运行时库并不只是标准库）。
2. `main` 函数是程序入口，但难道可执行文件的机器码一打开就是它吗？这需要有一个复杂的启动流程，是个从 `_start` 开始的兔子洞。
3. `printf` 是运行时库提供的符号。可这里难道不是直接调操作系统的 API 吗？实际上不管是 OS 的系统调用还是汇编指令，它们都不方便让你直接把字符串画到终端上，这些过程也要靠标准库帮你封装一下。

在缺少操作系统和标准库的裸机环境下（例如 Rust 的 [no_std](https://link.zhihu.com/?target=https%3A//docs.rust-embedded.org/book/intro/no-std.html)），上面的代码是跑不起来的。而这里的 stdio 只是标准库的冰山一角，再举几个非常常见的例子：

- 负责数学运算的 `math.h`：很多精简指令集或嵌入式的低端 CPU 未必会提供做 sin 和 cos 这类[三角函数](https://www.zhihu.com/search?q=三角函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2133648600})运算的指令，这时它们需要软件实现。
- 负责字符串的 `string.h`：你觉得硬件和操作系统会内置「比较字符串长度」这种功能吗？当然也是靠软件实现啦。

换句话说，虽然 C 的 if、for 和函数等语言特性都可以很朴素且优雅地映射（lowering）到汇编，但必然会有些没法直接映射到系统调用和汇编指令的常用功能，比如上面介绍的那几项。对于这些脏活累活，它们就需要由运行时库（例如 Linux 上的 glibc 和 Windows 上的 CRT）来实现。

> 如果你熟悉 JavaScript 但还不熟悉 C，我还有篇讲「[C 手动内存管理基础入门](https://zhuanlan.zhihu.com/p/356214452)」的教程应该适合你。

我们可以把「应用程序、运行时库和 OS」三者间的关系大致按这样来理解：

![img](https://pic1.zhimg.com/80/v2-188fa171a5d9086a08a0414cb94acc05_720w.jpg?source=1940ef5c)

注意运行时库并不只是标准库，你就算不显式 include 任何标准库，也有一些额外的代码会被编译器插入到最后的可执行文件里。比如上面提到的 main 函数，它在真正执行前就需要大量来自运行时库的辅助，一图胜千言（具体细节推荐参考 [Linux x86 Program Start Up](https://link.zhihu.com/?target=http%3A//dbp-consulting.com/tutorials/debugging/linuxProgramStartup.html)）：

![img](https://pic1.zhimg.com/80/v2-ac9b094dbe8f231765e32a0c79e204ef_720w.jpg?source=1940ef5c)

除了加载和退出这些程序必备的地方以外，运行时库还可以起到类似前端社区 polyfill 的作用，在程序执行过程中被隐式而「按需」地调用。例如 gcc 的 [libgcc](https://link.zhihu.com/?target=https%3A//gcc.gnu.org/onlinedocs/gccint/Libgcc.html) 和 clang 的 [compiler-rt](https://link.zhihu.com/?target=https%3A//compiler-rt.llvm.org/)（后者还被移植成了 Rust 的 [compiler-builtins](https://link.zhihu.com/?target=https%3A//github.com/rust-lang/compiler-builtins) ），这些库都是特定于编译器的，我们一般比较少听到，但其实也很好理解。的意思大概就是 「运行期所必需的东西」。