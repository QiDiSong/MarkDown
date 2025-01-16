mlir-tblgen -gen-dialect-decls include/compass.td -I /project/ai/scratch01/qidson01/code/llvm-project/mlir/include -o include/Dialect.h.inc --dialect compass
mlir-tblgen -gen-dialect-defs include/compass.td -I /project/ai/scratch01/qidson01/code/llvm-project/mlir/include -o src/Dialect.cpp.inc --dialect compass

mlir-tblgen -gen-op-decls include/compass.td -I /project/ai/scratch01/qidson01/code/llvm-project/mlir/include -o include/Op.h.inc --dialect compass
mlir-tblgen -gen-op-defs include/compass.td -I /project/ai/scratch01/qidson01/code/llvm-project/mlir/include -o src/Op.cpp.inc --dialect compass
