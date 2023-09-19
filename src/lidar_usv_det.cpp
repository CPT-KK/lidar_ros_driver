/***************************************************************************************************************************
@file:src/lidar_usv_det.cpp
@author:ljn
@date:2021.12
@brief: detect the usv via lidar sensor
***************************************************************************************************************************/
// ROS
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"
#include <sensor_msgs/Imu.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseArray.h>
#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>
// Eigen 几何模块
#include <Eigen/Geometry>
// pcl 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

// clustering
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// local
#include "tic_toc.h"


using namespace std;

ros::Publisher lidar_pub, targets_pub;
ros::Subscriber imu_sub;
double m_usv_length = 0.0;
double m_usv_width = 0.0;
int m_usv_intensity = 0;
// 包含功能1：remove the usv body from the lidar pointscloud


std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs;

pcl::PointCloud<pcl::PointXYZI>::Ptr pc;

#ifdef USE_IMU
bool label_imu_sub = false;
Eigen::Quaterniond imu_pose;
void imuCallback(const sensor_msgs::Imu & msg)
{
    label_imu_sub = true;
    imu_pose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
}
#endif


void lidarcallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2)
{
    TicToc timer;
    timer.tic();

    static int counter = 0;
    std::cout << "Processing"  << counter ++ << std::endl;


    // for (int i = 0; i < (int) lidar0->fields.size(); ++i) 
    // {
    //     std::cout << "lidar0: " << lidar0->fields[i].name << std::endl;
    // }
    // for (int i = 0; i < (int) lidar1->fields.size(); ++i) 
    // {
    //     std::cout << "lidar1: " << lidar1->fields[i].name << std::endl;
    // }
    // for (int i = 0; i < (int) lidar2->fields.size(); ++i) 
    // {
    //     std::cout << "lidar2: " << lidar2->fields[i].name << std::endl;
    // }

    pcl::fromROSMsg(*lidar0, *pcs[0]);
    pcl::fromROSMsg(*lidar1, *pcs[1]);
    pcl::fromROSMsg(*lidar2, *pcs[2]);
    std::cout << "Processed from msgs " << timer.toc()<< std::endl;  
    // *pc += *pc0;
    // *pc += *pc1;
    // *pc += *pc2;
    pc->reserve(pcs[0]->size() + pcs[1]->size() + pcs[2]->size());


    #pragma omp parallel for num_threads(8)
    for(int i = 0; i < pcs.size(); i++)
    {
        for(int j=0; j<pcs[i]->size(); j++)
        {
            if(pcs[i]->points[j].intensity < m_usv_intensity)
                continue;

            Eigen::Vector3d point(pcs[i]->points[j].x, pcs[i]->points[j].y, pcs[i]->points[j].z);
            if(point.x() > -m_usv_length && point.x() < m_usv_length && point.y() > -m_usv_width && point.y() < m_usv_width)
                continue;

            pcl::PointXYZI point_pcl = pcs[i]->points[j];
            #ifdef USE_IMU
            point = imu_pose * point;
            point_pcl.x = point.x();
            point_pcl.y = point.y();
            point_pcl.z = point.z();
            #endif
            pc->push_back(point_pcl);
        }
    }
    std::cout << "Processed filter " << timer.toc()<< std::endl;  

    // pcl::StatisticalOutlierRemoval<pcl::PointXYZI> outlierRemovalFilter;
    // outlierRemovalFilter.setInputCloud (pc);
    // outlierRemovalFilter.setMeanK (10);
    // outlierRemovalFilter.setStddevMulThresh (1.0);
    // outlierRemovalFilter.filter (*pc);
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlierRemovalFilter;
    outlierRemovalFilter.setInputCloud(pc);
    outlierRemovalFilter.setRadiusSearch(4);
    outlierRemovalFilter.setMinNeighborsInRadius (6);
    outlierRemovalFilter.filter (*pc);
    std::cout << "OutlierRemoval filter " << timer.toc()<< std::endl; 





    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud (pc);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance (2.0); // 2m
    ec.setMinClusterSize (5);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pc);
    ec.extract (cluster_indices);

    geometry_msgs::PoseArray targets;
    targets.header.stamp = lidar0->header.stamp;
    targets.header.frame_id = "USV_FLU";
    for (const auto& cluster : cluster_indices)
    {
        double x_sum = 0;
        double y_sum = 0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& idx : cluster.indices) {
            x_sum += (*pc)[idx].x;
            y_sum += (*pc)[idx].y;
            cloud_cluster->push_back((*pc)[idx]);
        }
        x_sum /= cluster.indices.size();
        y_sum /= cluster.indices.size();
        geometry_msgs::Pose target_pose;
        target_pose.position.x = x_sum;
        target_pose.position.y = y_sum;
        target_pose.position.z = 0;
        targets.poses.push_back(target_pose);
    }
    targets_pub.publish(targets);

    sensor_msgs::PointCloud2 pc_pub;
    pcl::toROSMsg(*pc, pc_pub);
    pc_pub.header.stamp = lidar0->header.stamp;
    pc_pub.header.frame_id = "USV_FLU";
    lidar_pub.publish(pc_pub);
    pc->clear();

    std::cout << "Processed " << timer.toc()<< std::endl;  
}


int main(int argc, char **argv) {
    
    ros::init(argc, argv, "lidar_usv_det");
    ros::NodeHandle nh("~");

    // Eigen::Matrix3d RotLidar1;
    // Eigen::Vector3d TransLidar1;
    // vector<double> lidar1ToBodyPosVec;
    // vector<double> lidar1ToBodyRotVec;
    // nh.param<vector<double>>("lidar1_to_body_pos", lidar1ToBodyPosVec, vector<double>());
    // nh.param<vector<double>>("lidar1_to_body_rot", lidar1ToBodyRotVec, vector<double>());
    // RotLidar1 = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(lidar1ToBodyRotVec.data(), 3, 3);
    // TransLidar1 = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(lidar1ToBodyPosVec.data(), 3, 1);

    lidar_pub = nh.advertise<sensor_msgs::PointCloud2>("/filter/lidar", 1);
    targets_pub = nh.advertise<geometry_msgs::PoseArray>("/filter/target", 1);
    
    nh.param<double>("usv_length", m_usv_length, 3.0);
    nh.param<double>("usv_width", m_usv_width, 1.2);
    nh.param<int>("usv_intensity", m_usv_intensity, 20);
    std::cout << m_usv_intensity << "," << m_usv_length << "," << m_usv_width << std::endl; 
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
