/*
 */

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cape_ros/Planes.h>
#include <cv_bridge/cv_bridge.h>
#include <cape_ros/cape.h>
#include <opencv2/opencv.hpp>


class CapeRosNode {

public:
  CapeRosNode(ros::NodeHandle nh);

  void simpleProjectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, Eigen::MatrixXf& cloud_array, const double& depth_cutoff);

  void organizePointCloudByCell(Eigen::MatrixXf& cloud_in, Eigen::MatrixXf& cloud_out, cv::Mat& cell_map);

  void overlaySegmentsOnImage(const cv::Mat& img, const cv::Mat &seg);

  void depthCallback(const sensor_msgs::ImagePtr &image);

  cape_ros::PlanesConstPtr generateMessage(const std_msgs::Header &header) const;

  void reset();

  void intensityCallback(const sensor_msgs::ImagePtr &image);

private:

  ros::Subscriber depth_sub;
  ros::Subscriber intensity_sub;
  ros::Publisher planes_pub_;
  ros::Publisher image_overlay_pub_;
  ros::NodeHandle nh_;

  const double DEPTH_SCALE_METERS = 0.001;
  bool intrinsics_ready = false;
  double fx, fy, cx, cy;
  double cos_angle_max_;
  double max_merge_dist_;
  double depth_cutoff_;
  bool cylinder_detection_ ;

  int patch_size_;
  int width_;
  int height_;

  cv::Mat depth_mat;
  cv::Mat X;
  cv::Mat Y;
  Eigen::MatrixXf cloud_array;
  Eigen::MatrixXf cloud_array_organized;

  vector<PlaneSeg> plane_params;
  vector<CylinderSeg> cylinder_params;


  cv::Mat X_pre;
  cv::Mat Y_pre;
  cv::Mat cell_map;
  cv::Mat seg_rz;
  cv::Mat seg_output;
  cv::Mat overlay_image;
  cv_bridge::CvImageConstPtr intensity_image_ptr_;

  std::shared_ptr<Cape> cape_extractor_;
};
