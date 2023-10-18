#pragma once
#include <string>
#include <numeric>


#include <pcl/common/common.h>


class orientation_calc
{
private:
  std::string criterion_;

public:
  orientation_calc(std::string value) : criterion_(value){};
  ~orientation_calc(){};
  bool LshapeFitting(const pcl::PointCloud<pcl::PointXYZI> &cluster, float &theta_optim);
  float calcClosenessCriterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
  float calc_var(const std::vector<float> &v);
  float calc_variances_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
  float calc_closeness_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
  float calc_area_criterion(const std::vector<float> &C_1, const std::vector<float> &C_2);
};


