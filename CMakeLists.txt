cmake_minimum_required(VERSION 3.2)

project(NNet)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# bin

add_executable("${PROJECT_NAME}_vgg" src_cpp/models/vgg.cpp)
add_executable("${PROJECT_NAME}_resnet" src_cpp/models/resnet.cpp)

# libtorch
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/lib/libtorch/)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_include_directories(${PROJECT_NAME}_vgg PRIVATE include)

target_link_libraries(${PROJECT_NAME}_vgg "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}_resnet "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME}_vgg PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME}_resnet PROPERTY CXX_STANDARD 14)
