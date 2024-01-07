#include "lshape_estimator.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <Eigen/Geometry>

bool orientation_calc::LshapeFitting(const pcl::PointCloud<pcl::PointXYZI> &cluster, float &theta_optim) {
    if (cluster.size() < 10)
        return false;

    /* Paper : IV2017, Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners */
    const float angle_reso = 0.5f * M_PI / 180.0f;
    const float max_angle = M_PI / 2.0f;
    float max_q = std::numeric_limits<float>::min();
    float q;
    std::vector<std::pair<float /*theta*/, float /*q*/>> Q;
    Q.reserve(500);

    // search
    Eigen::Vector2f e_1;        // col.3, Algo.2
    Eigen::Vector2f e_2;        // col.4, Algo.2
    std::vector<float> C_1;     // col.5, Algo.2
    std::vector<float> C_2;     // col.6, Algo.2
    C_1.reserve(2000);
    C_2.reserve(2000);
    for (float theta = 0; theta < max_angle; theta += angle_reso) {
        e_1 << std::cos(theta), std::sin(theta);
        e_2 << -std::sin(theta), std::cos(theta);
        // Eigen::Vector2f e_1(std::cos(theta), std::sin(theta));      // col.3, Algo.2
        // Eigen::Vector2f e_2(-std::sin(theta), std::cos(theta));     // col.4, Algo.2
        // std::vector<float> C_1;                   // col.5, Algo.2
        // std::vector<float> C_2;                   // col.6, Algo.2
        C_1.clear();
        C_2.clear();
        for (const auto &point : cluster) {
            C_1.push_back(point.x * e_1.x() + point.y * e_1.y());
            C_2.push_back(point.x * e_2.x() + point.y * e_2.y());
        }

        if (criterion_ == "AREA") {
            q = calc_area_criterion(C_1, C_2);
        } else if (criterion_ == "CLOSENESS") {
            q = calc_closeness_criterion(C_1, C_2);
        } else if (criterion_ == "VARIANCE") {
            q = calc_variances_criterion(C_1, C_2);
        } else {
            std::cout << "L Shaped Algorithm Criterion Is Not Supported." << std::endl;
            break;
        }
        Q.push_back(std::make_pair(theta, q));  // col.8, Algo.2
    }

    for (size_t i = 0; i < Q.size(); ++i) {
        if (Q[i].second > max_q || i == 0) {
            max_q = Q[i].second;
            theta_optim = Q[i].first;
        }
    }
    return true;
}

float orientation_calc::calc_area_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2) {
    const float c1_min = *std::min_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float c1_max = *std::max_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float c2_min = *std::min_element(C_2.begin(), C_2.end());  // col.3, Algo.4
    const float c2_max = *std::max_element(C_2.begin(), C_2.end());  // col.3, Algo.4

    float alpha = -(c1_max - c1_min) * (c2_max - c2_min);

    return alpha;
}

float orientation_calc::calc_closeness_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2) {
    // Paper : Algo.4 Closeness Criterion
    const float min_c_1 = *std::min_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float max_c_1 = *std::max_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float min_c_2 = *std::min_element(C_2.begin(), C_2.end());  // col.3, Algo.4
    const float max_c_2 = *std::max_element(C_2.begin(), C_2.end());  // col.3, Algo.4

    std::vector<float> D_1;  // col.4, Algo.4
    D_1.reserve(2000);
    for (const auto &c_1_element : C_1) {
        const float v = std::min(std::fabs(max_c_1 - c_1_element), std::fabs(c_1_element - min_c_1));
        D_1.push_back(v);
    }

    std::vector<float> D_2;  // col.5, Algo.4
    D_2.reserve(2000);
    for (const auto &c_2_element : C_2) {
        const float v = std::min(std::fabs(max_c_2 - c_2_element), std::fabs(c_2_element - min_c_2));
        D_2.push_back(v);
    }

    float beta = 0.0f;
    for (size_t i = 0; i < D_1.size(); i++) {
        float d = std::max(std::min(D_1[i], D_2[i]), 0.05f);
        beta += (1.0f / d);
    }
    return beta;
}

float orientation_calc::calc_variances_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2) {
    const float c1_min = *std::min_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float c1_max = *std::max_element(C_1.begin(), C_1.end());  // col.2, Algo.4
    const float c2_min = *std::min_element(C_2.begin(), C_2.end());  // col.3, Algo.4
    const float c2_max = *std::max_element(C_2.begin(), C_2.end());  // col.3, Algo.4

    std::vector<float> d1;  // col.4, Algo.4
    d1.reserve(2000);
    for (const auto &c_1_element : C_1) {
        const float v = std::min(std::fabs(c1_max - c_1_element), std::fabs(c_1_element - c1_min));
        d1.push_back(v);
    }

    std::vector<float> d2;  // col.5, Algo.4
    d2.reserve(2000);
    for (const auto &c_2_element : C_2) {
        const float v = std::min(std::fabs(c2_max - c_2_element), std::fabs(c_2_element - c2_min));
        d2.push_back(v);
    }

    std::vector<float> e1;
    std::vector<float> e2;
    assert(d1.size() == d2.size());
    e1.reserve(d1.size()+10);
    e2.reserve(d2.size()+10);
    for (size_t i = 0; i < d1.size(); i++) {
        if (d1[i] < d2[i]) {
            e1.push_back(d1[i]);
        } else {
            e2.push_back(d2[i]);
        }
    }

    float v1 = 0.0f;
    if (!e1.empty()) {
        v1 = calc_var(e1);
    }

    float v2 = 0.0f;
    if (!e2.empty()) {
        v2 = calc_var(e2);
    }

    return -v1 - v2;
}

float orientation_calc::calc_var(const std::vector<float> &v) {
    float sum = std::accumulate(std::begin(v), std::end(v), 0.0f);
    float mean = sum / v.size();
    float acc_var_num = 0.0f;

    std::for_each(std::begin(v), std::end(v), [&](const float d) { acc_var_num += (d - mean) * (d - mean); });

    return sqrt(acc_var_num / (v.size() - 1));
}
