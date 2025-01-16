TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt



SOURCES += main.cpp \
    src/Compass.cpp \
    src/Dialect.cpp \
    src/MLIRGen.cpp

HEADERS += \
    include/AST.h \
    include/Dialect.h \
    include/Lexer.h \
    include/MLIRGen.h \
    include/Parser.h


INCLUDEPATH += /project/ai/scratch01/qidson01/code/llvm-project/mlir/include
INCLUDEPATH += /project/ai/scratch01/qidson01/code/Compass_Unified_Parser/mlir/compass2mlir/include


LIBS += /project/ai/scratch01/qidson01/code/llvm-project/build/lib/libLLVMSupport.a
LIBS += /project/ai/scratch01/qidson01/code/llvm-project/build/lib/libMLIRIR.a
LIBS += /project/ai/scratch01/qidson01/code/llvm-project/build/lib/libMLIRSupport.a
LIBS += /project/ai/scratch01/qidson01/code/llvm-project/build/lib/libMLIRFuncDialect.a

DISTFILES += \
    gen_dialect_op.sh \
    include/Dialect.h.inc \
    include/Op.h.inc \
    include/compass.td \
    src/Dialect.cpp.inc \
    src/Op.cpp.inc \
    mobilenet_v2.bin \
    mobilenet_v2.txt
