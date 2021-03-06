/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#ifndef CAPE_ROS_CYLINDER_SEG_H
#define CAPE_ROS_CYLINDER_SEG_H

#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cape_ros/plane_seg.h>

using namespace std;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

class CylinderSeg {
 public:
  int nr_segments;
  vector<float> radii;
  double axis[3];
  vector<Eigen::MatrixXd> centers;
  vector<MatrixXb> inliers;
  vector<double> MSEs;
  int* local2global_map;
  vector<bool> cylindrical_mask;
  vector<Eigen::Vector3f> P1;
  vector<Eigen::Vector3f> P2;
  vector<float> P1P2_norm;
  CylinderSeg() {}
  CylinderSeg(vector<PlaneSeg*>& Grid, bool* activated_mask, int nr_cells_activated);
  float distance(Eigen::Vector3f& point, int segment_id);
  ~CylinderSeg(void);
};

#endif  // CAPE_ROS_CYLINDER_SEG_H
