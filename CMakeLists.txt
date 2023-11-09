cmake_minimum_required(VERSION 3.8)
project(lidar_ros_driver)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread -fopenmp")

find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        # msgs
        pcl_conversions
        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        visualization_msgs
        message_filters
)

catkin_package(
        CATKIN_DEPENDS
        std_msgs
        nav_msgs
        geometry_msgs
        sensor_msgs
        message_runtime
        pcl_conversions
        message_generation
        visualization_msgs
        message_filters
)
# find_package(OpenCV 3.2.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED)
# include directories
include_directories(
    ${catkin_INCLUDE_DIRS}
    include 
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PCL_CONVERSION_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

# link directories
link_directories(
        include
        ${PCL_LIBRARY_DIRS}
)

###########
## Build ##
###########
# Range Image Projection
add_executable(lidar_usv_det src/lidar_usv_det.cpp src/lshape_estimator.cpp)
add_dependencies(lidar_usv_det ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_usv_det ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${EIGEN3_LIBS} ${OpenCV_LIBS})
