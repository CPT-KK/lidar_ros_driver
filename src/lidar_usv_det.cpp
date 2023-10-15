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

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs;

pcl::PointCloud<pcl::PointXYZI>::Ptr pc;

#ifdef USE_IMU
bool label_imu_sub = false;
Eigen::Quaterniond imu_pose;

void calculateDimPos(const pcl::PointCloud<pcl::PointXYZI>& cluster, geometry_msgs::Pose& output, double& theta_star) {
    constexpr double ep = 0.001;
    // calc centroid point for cylinder height(z)
    pcl::PointXYZI centroid;
    centroid.x = 0;
    centroid.y = 0;
    centroid.z = 0;
    for (const auto& pcl_point : cluster) {
        centroid.x += pcl_point.x;
        centroid.y += pcl_point.y;
        centroid.z += pcl_point.z;
    }
    centroid.x = centroid.x / (double)cluster.size();
    centroid.y = centroid.y / (double)cluster.size();
    centroid.z = centroid.z / (double)cluster.size();

    // // calc min and max z for cylinder length
    double min_z = 0;
    double max_z = 0;
    for (size_t i = 0; i < cluster.size(); ++i) {
        if (cluster.at(i).z < min_z || i == 0)
            min_z = cluster.at(i).z;
        if (max_z < cluster.at(i).z || i == 0)
            max_z = cluster.at(i).z;
    }

    // calc circumscribed circle on x-y plane
    cv::Mat_<float> cv_points((int)cluster.size(), 2);
    for (size_t i = 0; i < cluster.size(); ++i) {
        cv_points(i, 0) = cluster.at(i).x;  // x
        cv_points(i, 1) = cluster.at(i).y;  // y
    }

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(cv::Mat(cv_points).reshape(2), center, radius);

    // Paper : Algo.2 Search-Based Rectangle Fitting
    Eigen::Vector2d e_1_star;  // col.11, Algo.2
    Eigen::Vector2d e_2_star;
    e_1_star << std::cos(theta_star), std::sin(theta_star);
    e_2_star << -std::sin(theta_star), std::cos(theta_star);
    std::vector<double> C_1_star;  // col.11, Algo.2
    std::vector<double> C_2_star;  // col.11, Algo.2
    for (const auto& point : cluster) {
        C_1_star.push_back(point.x * e_1_star.x() + point.y * e_1_star.y());
        C_2_star.push_back(point.x * e_2_star.x() + point.y * e_2_star.y());
    }

    // col.12, Algo.2
    const double min_C_1_star = *std::min_element(C_1_star.begin(), C_1_star.end());
    const double max_C_1_star = *std::max_element(C_1_star.begin(), C_1_star.end());
    const double min_C_2_star = *std::min_element(C_2_star.begin(), C_2_star.end());
    const double max_C_2_star = *std::max_element(C_2_star.begin(), C_2_star.end());

    const double a_1 = std::cos(theta_star);
    const double b_1 = std::sin(theta_star);
    const double c_1 = min_C_1_star;
    const double a_2 = -1.0 * std::sin(theta_star);
    const double b_2 = std::cos(theta_star);
    const double c_2 = min_C_2_star;
    const double a_3 = std::cos(theta_star);
    const double b_3 = std::sin(theta_star);
    const double c_3 = max_C_1_star;
    const double a_4 = -1.0 * std::sin(theta_star);
    const double b_4 = std::cos(theta_star);
    const double c_4 = max_C_2_star;

    // calc center of bounding box
    double intersection_x_1 = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
    double intersection_y_1 = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);
    double intersection_x_2 = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
    double intersection_y_2 = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

    // calc dimention of bounding box
    Eigen::Vector2d e_x;
    Eigen::Vector2d e_y;
    e_x << a_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1)), b_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1));
    e_y << a_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2)), b_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2));
    Eigen::Vector2d diagonal_vec;
    diagonal_vec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

    // calc yaw
    tf2::Quaternion quat;
    quat.setEuler(/* roll */ 0, /* pitch */ 0, /* yaw */ std::atan2(e_1_star.y(), e_1_star.x()));
    // std::cout << "yaw: " << std::atan2(e_1_star.y(), e_1_star.x()) << std::endl;

    output.orientation.w = quat.w();
    output.orientation.x = quat.x();
    output.orientation.y = quat.y();
    output.orientation.z = quat.z();
    // constexpr double ep = 0.001;
    // output.dimensions.x = std::fabs(e_x.dot(diagonal_vec));
    // output.dimensions.y = std::fabs(e_y.dot(diagonal_vec));
    // output.dimensions.z = std::max((max_z - min_z), ep);
    // output.pose_reliable = true;
    output.position.x = (intersection_x_1 + intersection_x_2) / 2.0;
    output.position.y = (intersection_y_1 + intersection_y_2) / 2.0;
    output.position.z = centroid.z;

    // check wrong output
    // output.dimensions.x = std::max(output.dimensions.x, ep);
    // output.dimensions.y = std::max(output.dimensions.y, ep);

    // changed by ljn
    // output.position.x = (intersection_x_1 + intersection_x_2) / 2.0; // x
    // output.position.y = (intersection_y_1 + intersection_y_2) / 2.0; // y
    // output.position.z = std::atan2(e_1_star.y(), e_1_star.x()); // yaw
    // output.orientation.x = std::fabs(e_x.dot(diagonal_vec)); // width
    // output.orientation.y = std::fabs(e_y.dot(diagonal_vec)); // length
    // output.orientation.z = std::max((max_z - min_z), ep); //height
    // // check wrong output
    // output.orientation.x = std::max(output.orientation.x, ep);
    // output.orientation.y = std::max(output.orientation.y, ep);
}

void cloudCut(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float xLB, float xUB, float yLB, float yUB, float zLB, float zUB) {
    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    // 定义裁剪对象
    pcl::PassThrough<pcl::PointXYZI> passX;
    pcl::PassThrough<pcl::PointXYZI> passY;
    pcl::PassThrough<pcl::PointXYZI> passZ;

    // 剪裁点云
    // x轴
    passX.setInputCloud(cloud);
    passX.setFilterFieldName("x");
    passX.setFilterLimits(xLB, xUB);  // 裁剪保留区域
    passX.filter(*cloud);

    // y轴
    passY.setInputCloud(cloud);
    passY.setFilterFieldName("y");
    passY.setFilterLimits(yLB, yUB);
    passY.filter(*cloud);

    // z轴
    passZ.setInputCloud(cloud);
    passZ.setFilterFieldName("z");
    passZ.setFilterLimits(zLB, zUB);
    passZ.filter(*cloud);

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
    pcl::fromROSMsg(*lidar0, *pcs[0]);
    pcl::fromROSMsg(*lidar1, *pcs[1]);
    pcl::fromROSMsg(*lidar2, *pcs[2]);
    ROS_INFO("Processed from msgs: %e s", (ros::Time::now() - t0).toSec()); 
    t0 = ros::Time::now();

    pc->reserve(pcs[0]->size() + pcs[1]->size() + pcs[2]->size());

    // Cut filter
#pragma omp parallel for num_threads(8)
    for (std::size_t i = 0; i < pcs.size(); i++) {
        for (std::size_t j = 0; j < pcs[i]->size(); j++) {
            if (pcs[i]->points[j].intensity < m_usv_intensity){
                continue;
            }
            
            if (pcs[i]->points[j].x > -m_usv_length && pcs[i]->points[j].x < m_usv_length && pcs[i]->points[j].y > -m_usv_width && pcs[i]->points[j].y < m_usv_width) {
                continue;
            }

            if (pcs[i]->points[j].z > 5.0f || pcs[i]->points[j].z < -2.0f) {
                continue;
            }

            if (abs(pcs[i]->points[j].x) > 120.0f || abs(pcs[i]->points[j].y) > 120.0f) {
                continue;
            }

            pcl::PointXYZI point_pcl = pcs[i]->points[j];

#ifdef USE_IMU
            Eigen::Vector3d thisPoint(pcs[i]->points[j].x, pcs[i]->points[j].y, pcs[i]->points[j].z);
            thisPoint = imu_pose * thisPoint;
            point_pcl.x = thisPoint[0];
            point_pcl.y = thisPoint[1];
            point_pcl.z = thisPoint[2];
#endif
            pc->push_back(point_pcl);
        }
    }
    ROS_INFO("Processed box filter: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Outlier filters
    cloudFilter(pc, 0.1f);
    ROS_INFO("Remove outliers: %e s", (ros::Time::now() - t0).toSec());
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
        float cenENUX = 0.0f;
        float cenENUY = 0.0f;
        clusterDirection(cloudCluster, objVector, len, wid, lwRatio, cenX, cenY);

        target_pose.position.x = cenX;
        target_pose.position.y = cenY;
        tf2::Quaternion quat;
        quat.setEuler(0, 0, std::atan(objVector[1]/objVector[0]));

        target_pose.orientation.w = quat.w();
        target_pose.orientation.x = quat.x();
        target_pose.orientation.y = quat.y();
        target_pose.orientation.z = quat.z();

        // double theta_optim = 0.0;
        // orientation_calc orient_calc_("AREA");  // AREA
        // bool success_fitting = orient_calc_.LshapeFitting(*cloudCluster, theta_optim);
        // calculateDimPos(*cloudCluster, target_pose, theta_optim);

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

#ifdef USE_IMU
    imu_sub = nh.subscribe("/mavros/imu/data", 1, imuCallback);
#endif

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar0(nh, "/livox/lidar_192_168_147_231", 3);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar1(nh, "/livox/lidar_192_168_147_232", 3);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar2(nh, "/livox/lidar_192_168_147_233", 3);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), sub_lidar0, sub_lidar1, sub_lidar2);
    sync.registerCallback(boost::bind(&lidarcallback, _1, _2, _3));

    pcs.resize(3);
    pcs[0].reset(new pcl::PointCloud<pcl::PointXYZI>());
    pcs[1].reset(new pcl::PointCloud<pcl::PointXYZI>());
    pcs[2].reset(new pcl::PointCloud<pcl::PointXYZI>());
    pc.reset(new pcl::PointCloud<pcl::PointXYZI>());

    ros::Rate rate(10.0);

    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
};
