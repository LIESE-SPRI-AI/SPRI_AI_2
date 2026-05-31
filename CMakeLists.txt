cmake_minimum_required(VERSION 3.15)

project(TestC)

find_package(OpenMP REQUIRED)

add_executable(app 
    src/main.c
    src/nn/tensor.c
    src/nn/conv2d.c
    src/nn/batchnorm2d.c
    src/nn/relu.c
    src/nn/doubleconv2d.c
    src/nn/maxpool2d.c
    src/nn/convtranspose2d.c
)

target_include_directories(app PRIVATE inc)

target_compile_options(app PRIVATE -O3)

target_link_libraries(app PRIVATE OpenMP::OpenMP_C)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")


#target_compile_options(app PRIVATE -S)

#Para compilar 

#### Debug 
#cmake -DCMAKE_BUILD_TYPE=Debug ..
#cmake --build .

#### Release

#cmake -DCMAKE_BUILD_TYPE=Release ..
#cmake --build .

