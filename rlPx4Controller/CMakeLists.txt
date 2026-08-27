cmake_minimum_required(VERSION 3.10)

# 项目名称
project(MyProject)

# 指定C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(my_executable src/controller_test.cpp)

# 如果有其他源文件，可以像下面这样添加它们
# add_executable(my_executable main.cpp other_source.cpp)

# 如果有其他头文件路径，可以像下面这样添加它们
target_include_directories(my_executable PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 如果有其他库依赖，可以像下面这样添加它们
# target_link_libraries(my_executable my_library)
