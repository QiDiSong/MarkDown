cmake_minimum_required(VERSION 3.16)
project(compass2mlir)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件路径
include_directories(
    include
    /project/ai/scratch01/qidson01/code/llvm-project/mlir/include
    /project/ai/scratch01/qidson01/code/llvm-project/llvm/include
    /project/ai/scratch01/qidson01/code/llvm-project/build/include
    /project/ai/scratch01/qidson01/code/llvm-project/build/tools/mlir/include/
)

# 添加源文件
set(SOURCES
    main.cpp
    src/Compass.cpp
    src/Dialect.cpp
    src/MLIRGen.cpp
)

set(HEADERS
    include/AST.h
    include/Dialect.h
    include/Lexer.h
    include/MLIRGen.h
    include/Parser.h
)

# 设置输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /project/ai/scratch01/qidson01/code/Compass_Unified_Parser/mlir/compass2mlir)

# 链接库路径
link_directories(
    /project/ai/scratch01/qidson01/code/llvm-project/build/lib
)

# 添加可执行文件
add_executable(compass2mlir ${SOURCES} ${HEADERS})

# 链接必要的 LLVM 和 MLIR 库
target_link_libraries(compass2mlir
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRParser
    MLIRExecutionEngine
    MLIRTransforms
    LLVMSupport
    LLVMCore
    LLVMDemangle
    LLVMMC
    LLVMBinaryFormat
    pthread
    dl
)

# 设置额外的文件（此部分仅用于项目管理，非必需）
set(DISTFILES
    gen_dialect_op.sh
    include/Dialect.h.inc
    include/Op.h.inc
    include/compass.td
    src/Dialect.cpp.inc
    src/Op.cpp.inc
    mobilenet_v2.bin
    mobilenet_v2.txt
)