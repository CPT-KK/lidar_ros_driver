#pragma once
#include <pcl/common/common.h>

#include <numeric>
#include <string>
#include <functional>

// #define CRITERION "AREA"
// #define CRITERION "CLOSENESS"
#define CRITERION "VARIANCE"

class OrientationCalc {
   private:
    const float angle_reso = 0.5f * M_PI / 180.0f;
    const float max_angle = M_PI / 2.0f;
    std::vector<float> theta_array;

    std::string criterion_;
    std::function<float(const std::vector<float> &, const std::vector<float> &)> calc_func_;

   public:
    OrientationCalc(std::string value) : criterion_(value){
        // Select method
        if (criterion_ == "AREA") {
            calc_func_ = std::bind(&OrientationCalc::calc_area_criterion, this, std::placeholders::_1, std::placeholders::_2);
        } else if (criterion_ == "CLOSENESS") {
            calc_func_ = std::bind(&OrientationCalc::calc_closeness_criterion, this, std::placeholders::_1, std::placeholders::_2);
        } else if (criterion_ == "VARIANCE") {
            calc_func_ = std::bind(&OrientationCalc::calc_variances_criterion, this, std::placeholders::_1, std::placeholders::_2);
        } else {
            throw std::invalid_argument("L Shaped Algorithm Criterion Is Not Supported.");
        }

        // Calculate theta array
        theta_array.reserve(100);
        for (float theta = 0; theta < max_angle; theta += angle_reso) {
            theta_array.push_back(theta);
        }
    };
    ~OrientationCalc(){};
    bool LshapeFitting(const pcl::PointCloud<pcl::PointXYZI> &cluster, float &theta_optim);
    float calc_var(const std::vector<float> &v);
    float calc_variances_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
    float calc_closeness_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
    float calc_area_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
};
