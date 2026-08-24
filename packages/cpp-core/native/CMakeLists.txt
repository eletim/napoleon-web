cmake_minimum_required(VERSION 3.16)

project(napoleon_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NAPOLEON_ENABLE_ONNXRUNTIME "Enable C++ ONNX Runtime policy backend" OFF)

add_library(napoleon_core
  src/napoleon_core.cpp
  src/napoleon_observation.cpp
  src/napoleon_joint_teacher.cpp
  src/napoleon_onnx_policy.cpp
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

if(NAPOLEON_ENABLE_ONNXRUNTIME)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "NAPOLEON_ENABLE_ONNXRUNTIME=ON requires onnxruntime_cxx_api.h and libonnxruntime")
  endif()
  target_compile_definitions(napoleon_core PRIVATE NAPOLEON_ENABLE_ONNXRUNTIME)
  target_include_directories(napoleon_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(napoleon_core PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

add_library(napoleon_observation INTERFACE)
target_link_libraries(napoleon_observation INTERFACE napoleon_core)

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

add_executable(napoleon_nonplaying_bidding_dataset_cli
  src/napoleon_nonplaying_bidding_dataset_cli.cpp
)

target_link_libraries(napoleon_nonplaying_bidding_dataset_cli PRIVATE napoleon_observation napoleon_core)
target_compile_options(napoleon_nonplaying_bidding_dataset_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_core_self_test
  src/napoleon_core_self_test.cpp
)

target_link_libraries(napoleon_core_self_test PRIVATE napoleon_core)
target_compile_options(napoleon_core_self_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_onnx_policy_smoke
  src/napoleon_onnx_policy_smoke.cpp
)

target_link_libraries(napoleon_onnx_policy_smoke PRIVATE napoleon_core)
target_compile_options(napoleon_onnx_policy_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_eval_cli
  src/napoleon_eval_cli.cpp
)

target_link_libraries(napoleon_eval_cli PRIVATE napoleon_core)
target_compile_options(napoleon_eval_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(napoleon_joint_teacher_cli
  src/napoleon_joint_teacher_cli.cpp
)

target_link_libraries(napoleon_joint_teacher_cli PRIVATE napoleon_core)
target_compile_options(napoleon_joint_teacher_cli PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME napoleon_core_self_test COMMAND napoleon_core_self_test)
