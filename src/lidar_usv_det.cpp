/***************************************************************************************************************************
@file:src/lidar_usv_det.cpp
@author:ljn
@date:2021.12
@brief: detect the usv via lidar sensor
***************************************************************************************************************************/
// ROS
#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <Eigen/Core>

#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Int32MultiArray.h"

// Eigen 几何模块
#include <Eigen/Geometry>
// pcl
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

// clustering
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/moment_of_inertia_estimation.h>

// opencv
#include "opencv2/opencv.hpp"

// local
#include "lshape_estimator.hpp"

#define USE_IMU
// #define EST_HEADING

using namespace std;

ros::Publisher lidar_pub, targets_pub;
ros::Subscriber imu_sub;
double m_usv_length = 0.0;
double m_usv_width = 0.0;
int m_usv_intensity = 0;
int counter = 0;

// 包含功能1：remove the usv body from the lidar pointscloud
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar1PC;
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar2PC;
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar3PC;
pcl::PointCloud<pcl::PointXYZI>::Ptr lidarRawPC;
pcl::PointCloud<pcl::PointXYZI>::Ptr pc;

#ifdef USE_IMU
bool label_imu_sub = false;
Eigen::Quaterniond imu_pose;

void cloudCut(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float xLB, float xUB, float yLB, float yUB, float zLB, float zUB,  float iLB, float iUB) {
    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    // 定义裁剪对象
    pcl::PassThrough<pcl::PointXYZI> pass;

    // 剪裁点云
    // x轴
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(xLB, xUB);  // 裁剪保留区域
    pass.filter(*cloud);

    // y轴
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(yLB, yUB);
    pass.filter(*cloud);

    // z轴
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(zLB, zUB);
    pass.filter(*cloud);

    // 强度
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("I");
    pass.setFilterLimits(iLB, iUB);
    pass.filter(*cloud);

    return;
}

void cloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float divFilter) {

    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    // 创建离群点滤波器对象
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;

    // 定义 VoxelGrid 滤波器变量
    pcl::VoxelGrid<pcl::PointXYZI> voxGrid;

    // 去除离群点
    sor.setInputCloud(cloud);

    // 设置在进行统计时考虑查询点邻近点数
    sor.setMeanK(100);

    // 设置判断是否为离群点的阈值
    sor.setStddevMulThresh(1.0f);

    // 执行滤波处理保存内点到inputCloud
    sor.filter(*cloud);

    // 叶素滤波
    voxGrid.setInputCloud(cloud);
    voxGrid.setLeafSize(divFilter, divFilter, divFilter);
    voxGrid.filter(*cloud);

    return;
}

std::vector<pcl::PointIndices> clusterExt(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    std::vector<pcl::PointIndices> clusterIndices;

    // 判断点云是否为空
    if (cloud->size() == 0) {
        return clusterIndices;
    }
    
    // 点云欧式聚类分割，基于 KdTree 对象作为搜索方法
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdTree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ecExtraction;
    ecExtraction.setClusterTolerance(2.0f);      // 点之间相隔大于10m的不被认为是一个聚类
    ecExtraction.setMinClusterSize(40);       // 一个聚类里所允许的最少点数
    ecExtraction.setMaxClusterSize(5000);    // 一个聚类里所允许的最多点数
    ecExtraction.setSearchMethod(kdTree);     // 设置搜索方法为 KdTreee
    ecExtraction.setInputCloud(cloud);     // 设置被搜索的点云

    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ecExtraction.extract(clusterIndices);

    return clusterIndices;
}

void clusterDirection(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float objVector[3], float& len, float& wid, float& lwRatio, float& cenX, float& cenY) {
    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    pcl::MomentOfInertiaEstimation<pcl::PointXYZI> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();

    std::vector<float> moment_of_inertia;
    std::vector<float> eccentricity;
    pcl::PointXYZI min_point_OBB;
    pcl::PointXYZI max_point_OBB;
    pcl::PointXYZI position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia(moment_of_inertia);                                      // 计算出的惯性矩
    feature_extractor.getEccentricity(eccentricity);                                              // 计算出的偏心率
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);  // OBB对应的相关参数
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);                     // 三个特征值
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);                 // 三个特征向量
    feature_extractor.getMassCenter(mass_center);                                                 // 计算质心

    for (int i = 0; i < 3; i++) {
        objVector[i] = major_vector[i];
    }
    
    len = max_point_OBB.x - min_point_OBB.x;
    wid = max_point_OBB.y - min_point_OBB.y;
    lwRatio = len/wid;
    cenX = mass_center[0];
    cenY = mass_center[1];

    // printf("Center: [%.2f, %.2f, %.2f] | Direction: [%.2f, %.2f, %.2f]\n", mass_center[0], mass_center[1], mass_center[2], major_vector[0], major_vector[1], major_vector[2]);

    return;
}

void imuCallback(const sensor_msgs::Imu& msg) {
    label_imu_sub = true;
    imu_pose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
}
#endif

void lidarcallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2) {
    // Record init time
    ros::Time t0 = ros::Time::now();
    ROS_INFO("Processing No.%d", counter++);

    // Merge 3 pointclouds
    pcl::fromROSMsg(*lidar0, *lidar1PC);
    pcl::fromROSMsg(*lidar1, *lidar2PC);
    pcl::fromROSMsg(*lidar2, *lidar3PC);
    *lidarRawPC = *lidar1PC + *lidar2PC + *lidar3PC;

    ROS_INFO("Processed from msgs: %e s", (ros::Time::now() - t0).toSec()); 
    t0 = ros::Time::now();

    // Cut filter
#pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < lidarRawPC->size(); i++) {
        // Intensity
        if(lidarRawPC->points[i].intensity < 10.0f) {
            continue;
        }

        // Outer box filter
        if (lidarRawPC->points[i].x < -200.0f || lidarRawPC->points[i].x > 200.0f) {
            continue;
        }
        if (lidarRawPC->points[i].y < -70.0f || lidarRawPC->points[i].y > 70.0f) {
            continue;
        }
        if (lidarRawPC->points[i].z < -1.5f || lidarRawPC->points[i].z > 5.0f) {
            continue;
        }

        // USV filter
        if (lidarRawPC->points[i].x < 3.6f && lidarRawPC->points[i].x > -3.6f && lidarRawPC->points[i].y < 1.25f && lidarRawPC->points[i].x > -1.25f) {
            continue;
        }

        // 现在的点应该是 OK 的
        lidarRawPC->points[i].z = 0.0f;
        Eigen::Vector3d thisPoint(lidarRawPC->points[i].x, lidarRawPC->points[i].y, lidarRawPC->points[i].z);
        thisPoint = imu_pose * thisPoint;

        pcl::PointXYZI point_pcl = lidarRawPC->points[i];
        point_pcl.x = thisPoint[0];
        point_pcl.y = thisPoint[1];
        point_pcl.z = thisPoint[2];
        pc->push_back(point_pcl);
    }

    ROS_INFO("Box filter costs: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Outlier filters
    cloudFilter(pc, 0.1f);
    ROS_INFO("Removing outliers costs: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Creating the KdTree object for the search method of the extraction
    std::vector<pcl::PointIndices> clusterIndices = clusterExt(pc);
    ROS_INFO("Extract clusters: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Prepare publish clusters
    geometry_msgs::PoseArray targets;
    targets.header.stamp = lidar0->header.stamp;
    targets.header.frame_id = "USV_FLU";
    targets.poses.reserve(clusterIndices.size());
#pragma omp parallel for num_threads(8)    
    for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
        // 创建临时保存点云簇的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZI>);

        // 通过下标，逐个填充
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
            cloudCluster->points.push_back(pc->points[*pit]);
        }

        // 提取点云位姿
        geometry_msgs::Pose target_pose;
        float objVector[3] = {0.0f, 0.0f, 0.0f};
        float len = 0.0f;
        float wid = 0.0f;
        float lwRatio = 0.0f;
        float cenX = 0.0f;
        float cenY = 0.0f;
        clusterDirection(cloudCluster, objVector, len, wid, lwRatio, cenX, cenY);

        if (len > 30.0f || wid > 10.0f) {
            continue;
        }

        target_pose.position.x = cenX;
        target_pose.position.y = cenY;
        tf2::Quaternion quat;
        quat.setEuler(0, 0, std::atan(objVector[1]/objVector[0]));

        target_pose.orientation.w = quat.w();
        target_pose.orientation.x = quat.x();
        target_pose.orientation.y = quat.y();
        target_pose.orientation.z = quat.z();
        targets.poses.push_back(target_pose);
    }
    targets_pub.publish(targets);
    ROS_INFO("Cluster state estimation finished in %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();
    
    
//     for (const auto& cluster : clusterIndices) {
//         pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
//         for (const auto& idx : cluster.indices) {
//             cloud_cluster->push_back((*pc)[idx]);
//         }

        
// #ifdef EST_HEADING  // Begin heading estimation
        
//         // Efficient L-shape fitting of laser scanner data for vehicle pose estimation
//         // Remove the outlier
//         orientation_calc orient_calc_("AREA");  // AREA
//         double theta_optim;
//         bool success_fitting = orient_calc_.LshapeFitting(*cloud_cluster, theta_optim);

//         if (!success_fitting) {
//             ROS_INFO("Fitting unsuccessful");
//             continue;
//         }

//         calculateDimPos(*cloud_cluster, target_pose, theta_optim);
 
// #endif // End heading estimation

//         targets.poses.push_back(target_pose);
//     }
//     targets_pub.publish(targets);

    // Send processed pointcloud
    sensor_msgs::PointCloud2 pc_pub;
    pcl::toROSMsg(*pc, pc_pub);
    pc_pub.header.stamp = lidar0->header.stamp;
    pc_pub.header.frame_id = "USV_FLU";
    lidar_pub.publish(pc_pub);
    pc->clear();

    ROS_INFO("Processed pointcloud published in %e s\n", (ros::Time::now() - t0).toSec());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_usv_det");
    ros::NodeHandle nh("~");

    lidar_pub = nh.advertise<sensor_msgs::PointCloud2>("/filter/lidar", 1);
    targets_pub = nh.advertise<geometry_msgs::PoseArray>("/filter/target", 1);

    nh.param<double>("usv_length", m_usv_length, 3.0);
    nh.param<double>("usv_width", m_usv_width, 1.2);
    nh.param<int>("usv_intensity", m_usv_intensity, 20);
    ROS_INFO("Params: Intensity: %d | X box filter: %.2f | Y box filter: %.2f", m_usv_intensity, m_usv_length, m_usv_width);

    imu_sub = nh.subscribe("/mavros/imu/data", 1, imuCallback);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar0(nh, "/livox/lidar_192_168_147_231", 3);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar1(nh, "/livox/lidar_192_168_147_232", 3);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar2(nh, "/livox/lidar_192_168_147_233", 3);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), sub_lidar0, sub_lidar1, sub_lidar2);
    sync.registerCallback(boost::bind(&lidarcallback, _1, _2, _3));

    // 预分配内存
    lidar1PC->points.reserve(10000);
    lidar2PC->points.reserve(10000);
    lidar3PC->points.reserve(10000);
    lidarRawPC->points.reserve(30000);
    pc->points.reserve(30000);

    ros::Rate rate(10.0);

    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
};
