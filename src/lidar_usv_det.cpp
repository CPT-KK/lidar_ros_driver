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
#include <pcl/filters/voxel_grid.h>
// opencv
#include "opencv2/opencv.hpp"

// local
#include "tic_toc.h"
#include "lshape_estimator.hpp"


#define USE_IMU
#define EST_HEADING

using namespace std;

ros::Publisher lidar_pub, targets_pub;
ros::Subscriber imu_sub;
double m_usv_length = 0.0;
double m_usv_width = 0.0;
double m_usv_height = 0.0;
double m_usv_downsample_factor = 0.0;
int m_usv_intensity = 0;
// 包含功能1：remove the usv body from the lidar pointscloud


std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcs;

pcl::PointCloud<pcl::PointXYZI>::Ptr pc;
pcl::VoxelGrid<pcl::PointXYZI> downsampleFilter;


#ifdef USE_IMU
bool label_imu_sub = false;
Eigen::Quaterniond imu_pose;



void calculateDimPos(const pcl::PointCloud<pcl::PointXYZI>& cluster, geometry_msgs::Pose& output, double &theta_star)
{
	constexpr double ep = 0.001;
	// calc centroid point for cylinder height(z)
	pcl::PointXYZI centroid;
	centroid.x = 0;
	centroid.y = 0;
	centroid.z = 0;
	for (const auto& pcl_point : cluster)
	{
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
	for (size_t i = 0; i < cluster.size(); ++i)
	{
	if (cluster.at(i).z < min_z || i == 0)
		min_z = cluster.at(i).z;
	if (max_z < cluster.at(i).z || i == 0)
		max_z = cluster.at(i).z;
	}

	// calc circumscribed circle on x-y plane
	cv::Mat_<float> cv_points((int)cluster.size(), 2);
	for (size_t i = 0; i < cluster.size(); ++i)
	{
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
	for (const auto& point : cluster)
	{
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
	std::cout << "yaw: " << std::atan2(e_1_star.y(), e_1_star.x()) * 180.0 / 3.14 << std::endl;
	std::cout << "yaw: " << std::atan2(e_1_star.y(), e_1_star.x()) * 180.0 / 3.14 << std::endl;
	std::cout << "yaw: " << std::atan2(e_1_star.y(), e_1_star.x()) * 180.0 / 3.14 << std::endl;

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

void imuCallback(const sensor_msgs::Imu & msg)
{
	label_imu_sub = true;

	imu_pose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
    Eigen::Vector3d eular0 = imu_pose.toRotationMatrix().eulerAngles(2, 1, 0);//ZYX
    //q1 yaw = 0
    auto q1_tf = tf::createQuaternionMsgFromRollPitchYaw(eular0[2], eular0[1], 0);
    imu_pose = Eigen::Quaterniond(q1_tf.w, q1_tf.x, q1_tf.y, q1_tf.z);
}
#endif


void lidarcallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2)
{
	TicToc timer;
	timer.tic();

	static int counter = 0;
	std::cout << "Processing "  << counter ++ << std::endl;


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
			if(point_pcl.z > m_usv_height)
				continue;
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
	
	// downsample 
	downsampleFilter.setInputCloud(pc);
	downsampleFilter.filter(*pc);


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
	std::cout << "cluster_indices " << timer.toc()<< std::endl; 

	geometry_msgs::PoseArray targets;
	targets.header.stamp = lidar0->header.stamp;
	targets.header.frame_id = "USV_FLU";
	for (const auto& cluster : cluster_indices)
	{

		pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
		for (const auto& idx : cluster.indices) {
			cloud_cluster->push_back((*pc)[idx]);
		}

		geometry_msgs::Pose target_pose;
		#ifdef EST_HEADING
		/* begin heading estimation */ 
		// Efficient L-shape fitting of laser scanner data for vehicle pose estimation
		// remove the outlier
		orientation_calc orient_calc_("AREA"); // AREA
		double theta_optim;
		bool success_fitting = orient_calc_.LshapeFitting(*cloud_cluster, theta_optim);

		if(!success_fitting)
		{
			ROS_INFO("Fitting unsuccessful");
			continue;
		}
			
		calculateDimPos(*cloud_cluster, target_pose, theta_optim);
	

		// end heading estimation
		#endif

		targets.poses.push_back(target_pose);
		std::cout << "cluster_indice      " << timer.toc()<< std::endl;
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
	nh.param<double>("usv_height", m_usv_height, 5.0);
	nh.param<double>("usv_downsample_factor", m_usv_downsample_factor, 5.0);
	nh.param<int>("usv_intensity", m_usv_intensity, 20);
	
	std::cout << m_usv_intensity << "," << m_usv_length << "," << m_usv_width << std::endl; 
	#ifdef USE_IMU
	imu_sub = nh.subscribe("/mavros/imu/data", 1, imuCallback);
	#endif

	message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar0(nh, "/livox/lidar_192_168_147_231", 1);
	message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar1(nh, "/livox/lidar_192_168_147_232", 1);
	message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar2(nh, "/livox/lidar_192_168_147_233", 1);

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), sub_lidar0, sub_lidar1, sub_lidar2);
	sync.registerCallback(boost::bind(&lidarcallback, _1, _2, _3));

	pcs.resize(3);
	pcs[0].reset(new pcl::PointCloud<pcl::PointXYZI>());
	pcs[1].reset(new pcl::PointCloud<pcl::PointXYZI>());
	pcs[2].reset(new pcl::PointCloud<pcl::PointXYZI>());
	pc.reset(new pcl::PointCloud<pcl::PointXYZI>());
	downsampleFilter.setLeafSize(m_usv_downsample_factor, m_usv_downsample_factor, m_usv_downsample_factor);

	ros::Rate rate(10.0);

	while (ros::ok()) {

		ros::spinOnce();
		rate.sleep();
	}
	return 0;
};
