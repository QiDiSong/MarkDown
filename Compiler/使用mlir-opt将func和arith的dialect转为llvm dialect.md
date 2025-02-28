如何使用mlir-opt将arith和func这两个dialect转为llvm，再试用mlir-cpu-runner进行运行 下面是我的mlir module {  func.func @simple_add() -> f32 {    %0 = arith.constant 2.0 : f32    %1 = arith.constant 3.0 : f32    %2 = arith.addf %0, %1 : f32    return %2 : f32  } }







要将你的MLIR代码从`arith`和`func` dialect转换为LLVM dialect，并使用`mlir-cpu-runner`运行，你可以按照以下步骤操作：

### 1. 将MLIR代码保存到文件

首先，将你的MLIR代码保存到一个文件中，例如`simple_add.mlir`。

mlir

复制

```
// simple_add.mlir
module {
  func.func @simple_add() -> f32 {
    %0 = arith.constant 2.0 : f32
    %1 = arith.constant 3.0 : f32
    %2 = arith.addf %0, %1 : f32
    return %2 : f32
  }
}
```

### 2. 使用`mlir-opt`进行转换

使用`mlir-opt`工具将`arith`和`func` dialect转换为LLVM dialect。你可以通过以下命令实现：

bash

复制

```
mlir-opt --convert-arith-to-llvm --convert-func-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --reconcile-unrealized-casts simple_add.mlir > simple_add_llvm.mlir
```

这个命令的作用是：

- `--convert-arith-to-llvm`：将`arith` dialect转换为LLVM dialect。
- `--convert-func-to-llvm`：将`func` dialect转换为LLVM dialect。
- `--convert-scf-to-cf`：将结构化控制流（SCF）转换为控制流（CF）。
- `--convert-cf-to-llvm`：将控制流（CF）转换为LLVM dialect。
- `--reconcile-unrealized-casts`：处理未实现的类型转换。

生成的`simple_add_llvm.mlir`文件将包含转换后的LLVM dialect代码。

### 3. 使用`mlir-cpu-runner`运行

接下来，你可以使用`mlir-cpu-runner`来运行转换后的LLVM dialect代码。假设你已经编译了MLIR并安装了相关的运行时库，可以运行以下命令：

bash

复制

```
mlir-cpu-runner --entry-point-result=f32 --shared-libs=/path/to/libmlir_runner_utils.so --shared-libs=/path/to/libmlir_c_runner_utils.so simple_add_llvm.mlir
```

这个命令的作用是：

- `--entry-point-result=f32`：指定入口函数的返回类型为`f32`。
- `--shared-libs`：指定运行时库的路径。你需要根据你的MLIR安装路径来指定这些库的位置。

### 4. 查看输出

如果一切顺利，`mlir-cpu-runner`将运行你的MLIR代码，并输出结果。对于这个简单的加法例子，你应该会看到输出`5.0`。

### 注意事项

- 确保你已经正确安装了MLIR，并且`mlir-opt`和`mlir-cpu-runner`在系统的`PATH`中。
- 运行时库的路径（`libmlir_runner_utils.so`和`libmlir_c_runner_utils.so`）需要根据你的MLIR安装路径进行调整。

通过以上步骤，你应该能够成功将`arith`和`func` dialect转换为LLVM dialect，并使用`mlir-cpu-runner`运行你的MLIR代码。













--shared-libs=/path/to/libmlir_runner_utils.so --shared-libs=/path/to/libmlir_c_runner_utils.so  这两个在哪找







`libmlir_runner_utils.so` 和 `libmlir_c_runner_utils.so` 是 MLIR 运行时库，它们通常是在构建 MLIR 项目时生成的。如果你已经构建了 MLIR，可以通过以下方式找到这些库：

------

### 1. **默认路径**

如果你使用的是 LLVM/MLIR 的官方构建流程，这些库通常位于构建目录的 `lib` 子目录中。假设你的构建目录是 `build`，那么路径可能是：

bash

复制

```
build/lib/libmlir_runner_utils.so
build/lib/libmlir_c_runner_utils.so
```

------

### 2. **如何找到构建目录**

如果你不确定构建目录在哪里，可以尝试以下方法：

#### 方法 1：查找 `mlir-opt` 的路径

运行以下命令找到 `mlir-opt` 的路径：

bash

复制

```
which mlir-opt
```

假设输出是 `/path/to/build/bin/mlir-opt`，那么库文件通常位于 `/path/to/build/lib/` 目录下。

#### 方法 2：查找 MLIR 的安装目录

如果你将 MLIR 安装到了某个目录（例如通过 `cmake --install`），可以检查安装目录下的 `lib` 文件夹：

bash

复制

```
/path/to/install/lib/libmlir_runner_utils.so
/path/to/install/lib/libmlir_c_runner_utils.so
```

------

### 3. **如果找不到这些库**

如果你找不到这些库，可能是因为你没有启用 MLIR 的运行时支持。你需要重新构建 MLIR 并确保启用了相关选项。

#### 重新构建 MLIR

1. 克隆 LLVM 和 MLIR 源码（如果还没有）：

   bash

   复制

   ```
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   ```

2. 创建构建目录并配置 CMake：

   bash

   复制

   ```
   mkdir build
   cd build
   cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="mlir" \
     -DLLVM_BUILD_EXAMPLES=ON \
     -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
     -DLLVM_TARGETS_TO_BUILD="host" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_RTTI=ON \
     -DMLIR_ENABLE_EXECUTION_ENGINE=ON
   ```

   关键选项：

   - `-DMLIR_ENABLE_EXECUTION_ENGINE=ON`：启用 MLIR 运行时支持。
   - `-DLLVM_BUILD_EXAMPLES=ON`：确保构建示例和运行时库。

3. 构建 MLIR：

   bash

   复制

   ```
   ninja
   ```

4. 构建完成后，库文件会出现在 `build/lib/` 目录下。

------

### 4. **使用相对路径**

如果你不想指定完整路径，可以将库文件所在的目录添加到 `LD_LIBRARY_PATH` 环境变量中，然后直接使用库文件名：

bash

复制

```
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
mlir-cpu-runner --entry-point-result=f32 --shared-libs=libmlir_runner_utils.so --shared-libs=libmlir_c_runner_utils.so simple_add_llvm.mlir
```

------

### 5. **验证库文件是否存在**

运行以下命令检查库文件是否存在：

bash

复制

```
ls /path/to/build/lib/libmlir_runner_utils.so
ls /path/to/build/lib/libmlir_c_runner_utils.so
```

如果文件存在，就可以在 `mlir-cpu-runner` 中使用它们。

------

### 总结

- 默认路径：`build/lib/libmlir_runner_utils.so` 和 `build/lib/libmlir_c_runner_utils.so`。
- 如果找不到，重新构建 MLIR 并确保启用了 `-DMLIR_ENABLE_EXECUTION_ENGINE=ON`。
- 可以通过设置 `LD_LIBRARY_PATH` 简化路径。

