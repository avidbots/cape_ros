/*
 * This file is part of the cape_ros.
 * Copyright (c) 2019 Avidbots Corp.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CAPE_ROS_CAPE_ROS_NODE_H
#define CAPE_ROS_CAPE_ROS_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cape_ros/Planes.h>
#include <cv_bridge/cv_bridge.h>
#include <cape_ros/cape.h>
#include <opencv2/opencv.hpp>

class CapeRosNode
{
public:
  CapeRosNode(ros::NodeHandle nh);

  void simpleProjectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, Eigen::MatrixXf& cloud_array,
                               const double& depth_cutoff);

  void organizePointCloudByCell(Eigen::MatrixXf& cloud_in, Eigen::MatrixXf& cloud_out, cv::Mat& cell_map);

  void overlaySegmentsOnImage(const cv::Mat& img, const cv::Mat& seg);

  void depthCallback(const sensor_msgs::ImagePtr& image);

  cape_ros::PlanesConstPtr generateMessage(const std_msgs::Header& header) const;

  void reset();

  void intensityCallback(const sensor_msgs::ImagePtr& image);

private:
  ros::NodeHandle nh_;
  ros::Subscriber depth_sub_;
  ros::Subscriber intensity_sub_;
  ros::Publisher planes_pub_;
  ros::Publisher image_overlay_pub_;

  const double depth_scale_ = 0.001;
  bool intrinsics_ready_ = false;
  double fx_, fy_, cx_, cy_;
  double cos_angle_max_;
  double max_merge_dist_;
  double depth_cutoff_;
  bool cylinder_detection_;

  int patch_size_;
  int width_;
  int height_;

  cv::Mat depth_mat_;
  cv::Mat X_;
  cv::Mat Y_;
  Eigen::MatrixXf cloud_array_;
  Eigen::MatrixXf cloud_array_organized_;

  vector<PlaneSeg> plane_params_;
  vector<CylinderSeg> cylinder_params_;

  cv::Mat X_pre_;
  cv::Mat Y_pre_;
  cv::Mat cell_map_;
  cv::Mat seg_rz_;
  cv::Mat seg_output_;
  cv::Mat overlay_image_;
  cv_bridge::CvImageConstPtr intensity_image_ptr_;

  std::shared_ptr<Cape> cape_extractor_;
};

#endif  // CAPE_ROS_CAPE_ROS_NODE_H
