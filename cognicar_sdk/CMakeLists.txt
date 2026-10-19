cmake_minimum_required(VERSION 3.16)
project(cognicar_sdk_examples LANGUAGES CXX)

# Build the SDK's C++ examples against the prebuilt libraries in this bundle:
#   cmake -S . -B build && cmake --build build -j
# ROS bundles select the ROS interface automatically. DirectUdp-only bundles
# use the plain interface and require no ROS installation.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SDK "${CMAKE_CURRENT_LIST_DIR}")
find_package(Threads REQUIRED)

set(CMAKE_BUILD_RPATH   "${SDK}/lib")
set(CMAKE_INSTALL_RPATH  "${SDK}/lib")

# ---- DirectUdp interface (always available) ---------------------------------
add_library(cognicar_interface SHARED IMPORTED)
set_target_properties(cognicar_interface PROPERTIES
  IMPORTED_LOCATION "${SDK}/lib/libcognicar_interface.so"
  INTERFACE_INCLUDE_DIRECTORIES "${SDK}/include")
target_link_libraries(cognicar_interface INTERFACE Threads::Threads)
set(COGNICAR_EXERCISE_INTERFACE cognicar_interface)

# Prefer the ROS interface when this is a normal ROS-enabled SDK.
if(EXISTS "${SDK}/lib/libcognicar_interface_ros.so")
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)

  add_library(cognicar_interface_ros SHARED IMPORTED)
  set_target_properties(cognicar_interface_ros PROPERTIES
    IMPORTED_LOCATION "${SDK}/lib/libcognicar_interface_ros.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK}/include")
  target_link_libraries(cognicar_interface_ros INTERFACE
    Threads::Threads rclcpp::rclcpp
    ${std_msgs_TARGETS} ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS}
    ${sensor_msgs_TARGETS})
  set(COGNICAR_EXERCISE_INTERFACE cognicar_interface_ros)
endif()

# ---- Reusable controller libraries -----------------------------------------
if(EXISTS "${SDK}/lib/libtrajectory_tracking_core.so")
  add_library(trajectory_tracking_core SHARED IMPORTED)
  set_target_properties(trajectory_tracking_core PROPERTIES
    IMPORTED_LOCATION "${SDK}/lib/libtrajectory_tracking_core.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK}/include")
endif()
if(EXISTS "${SDK}/lib/liblow_level_control_core.so")
  add_library(low_level_control_core SHARED IMPORTED)
  set_target_properties(low_level_control_core PROPERTIES
    IMPORTED_LOCATION "${SDK}/lib/liblow_level_control_core.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK}/include")
endif()
if(EXISTS "${SDK}/lib/libstate_estimator_core.so")
  find_package(Eigen3 REQUIRED NO_MODULE)
  add_library(state_estimator_core SHARED IMPORTED)
  set_target_properties(state_estimator_core PROPERTIES
    IMPORTED_LOCATION "${SDK}/lib/libstate_estimator_core.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK}/include")
  target_link_libraries(state_estimator_core INTERFACE Eigen3::Eigen)
endif()

# Two kinds of program, one rule each:
#   exercises/*.cpp     main programs; all exercise dirs are include paths
#   exercises/*/*.cpp   one self-contained program per exercise
file(GLOB exercise_include_dirs LIST_DIRECTORIES true "${SDK}/exercises/*")
foreach(candidate_dir ${exercise_include_dirs})
  if(NOT IS_DIRECTORY "${candidate_dir}")
    list(REMOVE_ITEM exercise_include_dirs "${candidate_dir}")
  endif()
endforeach()

function(add_exercise_program exercise_src)
  get_filename_component(exercise_name "${exercise_src}" NAME_WE)
  add_executable(${exercise_name} "${exercise_src}")
  target_include_directories(${exercise_name} PRIVATE ${ARGN})
  target_compile_definitions(${exercise_name} PRIVATE COGNICAR_SDK_DIR="${SDK}")
  target_link_libraries(${exercise_name} PRIVATE ${COGNICAR_EXERCISE_INTERFACE})
  if(TARGET trajectory_tracking_core)
    target_link_libraries(${exercise_name} PRIVATE trajectory_tracking_core)
  endif()
  if(TARGET low_level_control_core)
    target_link_libraries(${exercise_name} PRIVATE low_level_control_core)
  endif()
  if(TARGET state_estimator_core)
    target_link_libraries(${exercise_name} PRIVATE state_estimator_core)
  endif()
endfunction()

file(GLOB main_sources "${SDK}/exercises/*.cpp")
foreach(main_src ${main_sources})
  add_exercise_program("${main_src}" ${exercise_include_dirs})
endforeach()

file(GLOB exercise_sources "${SDK}/exercises/*/*.cpp")
foreach(exercise_src ${exercise_sources})
  get_filename_component(exercise_dir "${exercise_src}" DIRECTORY)
  add_exercise_program("${exercise_src}" "${exercise_dir}")
endforeach()
