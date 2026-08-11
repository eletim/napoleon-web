cmake_minimum_required(VERSION 3.16)

project(napoleon_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(napoleon_core
  src/napoleon_core.cpp
  src/napoleon_evaluation.cpp
  src/napoleon_roster.cpp
  src/napoleon_rule_based.cpp
  src/napoleon_simulation_runtime.cpp
)

target_include_directories(napoleon_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(napoleon_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(napoleon_observation
  src/napoleon_observation.cpp
)

target_link_libraries(napoleon_observation PUBLIC napoleon_core)
target_compile_options(napoleon_observation PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_core_cli
  src/napoleon_core_cli.cpp
)

target_link_libraries(napoleon_core_cli PRIVATE napoleon_observation)
target_compile_options(napoleon_core_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_rl_dataset_cli
  src/napoleon_rl_dataset_cli.cpp
)

target_link_libraries(napoleon_rl_dataset_cli PRIVATE napoleon_observation napoleon_core)
target_compile_options(napoleon_rl_dataset_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_core_self_test
  src/napoleon_core_self_test.cpp
)

target_link_libraries(napoleon_core_self_test PRIVATE napoleon_core)
target_compile_options(napoleon_core_self_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_eval_cli
  src/napoleon_eval_cli.cpp
)

target_link_libraries(napoleon_eval_cli PRIVATE napoleon_core)
target_compile_options(napoleon_eval_cli PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME napoleon_core_self_test COMMAND napoleon_core_self_test)
