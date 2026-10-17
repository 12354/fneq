cmake_minimum_required(VERSION 3.21)
project(fneq-lift C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(remill REQUIRED)
find_package(LLVM REQUIRED CONFIG)

add_executable(fneq-lift fneq_lift.cpp)
target_include_directories(fneq-lift SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFS NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(fneq-lift PRIVATE ${LLVM_DEFS})
target_link_libraries(fneq-lift PRIVATE remill)
install(TARGETS fneq-lift RUNTIME DESTINATION bin)
