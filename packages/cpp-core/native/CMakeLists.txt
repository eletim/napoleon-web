cmake_minimum_required(VERSION 3.16)

project(napoleon_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(napoleon_core
  src/napoleon_core.cpp
)

target_include_directories(napoleon_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(napoleon_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_core_cli
  src/napoleon_core_cli.cpp
)

target_link_libraries(napoleon_core_cli PRIVATE napoleon_core)
target_compile_options(napoleon_core_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_core_self_test
  src/napoleon_core_self_test.cpp
)

target_link_libraries(napoleon_core_self_test PRIVATE napoleon_core)
target_compile_options(napoleon_core_self_test PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME napoleon_core_self_test COMMAND napoleon_core_self_test)
