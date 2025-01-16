`mlir-tblgen` 是 MLIR 的一个工具，默认是 LLVM 项目的一部分，因此你需要通过构建 LLVM/MLIR 来安装它。以下是安装和构建 `mlir-tblgen` 的详细步骤：

------

### **1. 下载 LLVM/MLIR 源代码**

`mlir-tblgen` 是 LLVM 项目的一个子组件，因此需要从 LLVM 官方仓库获取源码。

```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

------

### **2. 配置构建环境**

创建一个独立的构建目录并配置 CMake 构建。

```
mkdir build
cd build
```

<font color='red'>运行以下命令来生成构建文件</font>：

<font color='red'>注意：cmake有缓存，如果上次使用GCC7.3，而这是你改变了GCC版本为9.3，如果不删掉build文件夹重新编译的话，默认还是使用上次GCC7.3的缓存</font>

```
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON  \
  -DBUILD_SHARED_LIBS=ON
```

-DBUILD_SHARED_LIBS=ON   用于生成共享库

-DLLVM_ENABLE_RTTI=ON   用于开启RTTI模块

#### **参数说明**：

- `-DLLVM_ENABLE_PROJECTS="mlir"`：表示只构建 MLIR。
- `-DLLVM_BUILD_EXAMPLES=ON`：构建示例程序（可选）。
- `-DLLVM_TARGETS_TO_BUILD="X86"`：构建目标体系架构（如需要ARM等，修改为对应架构）。
- `-DCMAKE_BUILD_TYPE=Release`：构建为 Release 版本。
- `-DLLVM_ENABLE_ASSERTIONS=ON`：启用断言（调试时有用）。

------

### **3. 构建项目**

使用 Ninja 构建工具来编译项目（推荐使用 Ninja 进行快速构建）：

```
ninja -j 64
```

#### **构建完成后**：

构建目录下的 `bin` 文件夹中将包含 `mlir-tblgen` 工具，路径类似于：

```
llvm-project/build/bin/mlir-tblgen
```

------

### **4. 配置环境变量**

将 `mlir-tblgen` 添加到环境变量中，确保可以全局访问。

```
export PATH=/path/to/llvm-project/build/bin:$PATH
```

将上述命令添加到 `~/.bashrc` 或 `~/.zshrc` 文件中，确保每次登录后自动加载。

------

### **5. 验证安装**

确保 `mlir-tblgen` 安装成功，运行以下命令检查：

```
mlir-tblgen --version
```

如果命令返回版本信息，则表示安装成功。

------

### **6. 直接安装预编译的 LLVM/MLIR（可选）**

如果你不想手动构建，可以尝试使用系统包管理器直接安装 LLVM 和 MLIR（如果支持）。

#### **Ubuntu (apt)**:

```
sudo apt update
sudo apt install llvm-dev
```

#### **macOS (Homebrew)**:

```
brew install llvm
```

安装完成后，检查是否包含 `mlir-tblgen` 工具：

```
which mlir-tblgen
```

------

### **7. 常见问题解决**

- **找不到 Ninja**： 安装 Ninja：

  ```
  sudo apt install ninja-build  # Ubuntu
  brew install ninja           # macOS
  ```

- **找不到 CMake**： 安装 CMake：

  ```
  sudo apt install cmake       # Ubuntu
  brew install cmake           # macOS
  ```

- **构建失败**：

  - 确保使用支持 C++17 的编译器（如 GCC 7+ 或 Clang 7+）。
  - 检查 CMake 配置日志，确保没有缺少依赖项。

------

通过上述步骤安装和配置好 `mlir-tblgen` 后，你就可以运行 `gen_dialect_op.sh` 脚本来生成 Dialect 和 Operation 的代码了。