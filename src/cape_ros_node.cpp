/*
 */

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cape_ros/Planes.h>
#include <cv_bridge/cv_bridge.h>
#include <cape_ros/cape.h>
#include <opencv2/opencv.hpp>
#include <cape_ros/cape_ros_node.h>

std::array<uint8_t, 3> hsvToRgb(const double h, const double s, const double v)
{
  const auto c = v * s;
  const auto h_prime = fmod(h / 60.0, 6);
  const auto x = c * (1 - std::abs(fmod(h_prime, 2) - 1));
  const auto m = v - c;

  const auto v1 = static_cast<uint8_t>(255 * (c + m));
  const auto v2 = static_cast<uint8_t>(255 * (x + m));
  const auto v3 = static_cast<uint8_t>(255 * m);

  if (h_prime < 1)
  {
    return { v1, v2, v3 };
  }
  if (h_prime < 2)
  {
    return { v2, v1, v3 };
  }
  if (h_prime < 3)
  {
    return { v3, v1, v2 };
  }
  if (h_prime < 4)
  {
    return { v3, v2, v1 };
  }
  if (h_prime < 5)
  {
    return { v2, v3, v1 };
  }
  if (h_prime < 6)
  {
    return { v1, v3, v2 };
  }

  return { v3, v3, v3 };
}

CapeRosNode::CapeRosNode(ros::NodeHandle nh)
  : nh_(nh)
  , cos_angle_max_(std::cos(M_PI / 30))
  , max_merge_dist_(50.0)
  , depth_cutoff_(5.0)
  , cylinder_detection_(false)
  , patch_size_(20)
{
  depth_sub = nh.subscribe("depth_in", 2, &CapeRosNode::depthCallback, this);
  intensity_sub = nh.subscribe("ir_in", 2, &CapeRosNode::intensityCallback, this);
  planes_pub_ = nh.advertise<cape_ros::Planes>("planes", 1);
  image_overlay_pub_ = nh.advertise<sensor_msgs::Image>("overlay_image", 1);
}

void CapeRosNode::simpleProjectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, Eigen::MatrixXf& cloud_array,
                                          const double& depth_cutoff)
{
  int width = X.cols;
  int height = X.rows;

  float *sz, *sx, *sy;
  int id;
  for (int r = 0; r < height; r++)
  {
    sx = X.ptr<float>(r);
    sy = Y.ptr<float>(r);
    sz = Z.ptr<float>(r);
    for (int c = 0; c < width; c++)
    {
      if (sz[c] < depth_cutoff)
      {
        id = r * width + c;
        cloud_array(id, 0) = sx[c];
        cloud_array(id, 1) = sy[c];
        cloud_array(id, 2) = sz[c];
      }
    }
  }
}

void CapeRosNode::organizePointCloudByCell(Eigen::MatrixXf& cloud_in, Eigen::MatrixXf& cloud_out, cv::Mat& cell_map)
{
  int width = cell_map.cols;
  int height = cell_map.rows;
  int mxn = width * height;
  int mxn2 = 2 * mxn;

  int id, it(0);
  int* cell_map_ptr;
  for (int r = 0; r < height; r++)
  {
    cell_map_ptr = cell_map.ptr<int>(r);
    for (int c = 0; c < width; c++)
    {
      id = cell_map_ptr[c];
      *(cloud_out.data() + id) = *(cloud_in.data() + it);
      *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
      *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
      it++;
    }
  }
}

void CapeRosNode::overlaySegmentsOnImage(const cv::Mat& img, const cv::Mat& seg)
{
  std::vector<std::array<uint8_t, 3>> colors;
  colors.reserve(plane_params_.size());

  const int hue_step = 360 / (plane_params_.size() - 1);
  const double saturation = 0.9;
  const double value = 0.8;
  for (auto i = 0; i < plane_params_.size(); ++i)
  {
    colors.emplace_back(hsvToRgb(static_cast<double>(i * hue_step), saturation, value));
  }

  overlay_image_.create(height_, width_, CV_8UC3);

  for (int r = 0; r < height_; ++r)
  {
    for (int c = 0; c < width_; ++c)
    {
      auto& overlay_color = overlay_image_.at<cv::Vec3b>(r, c);
      const auto seg_code = seg.at<uint8_t>(r, c);
      const auto intensity = img.at<uint8_t>(r, c);

      if (seg_code)
      {
        const auto color = colors[seg_code - 1];
        overlay_color[0] = (intensity + color[0]) >> 1;
        overlay_color[1] = (intensity + color[1]) >> 1;
        overlay_color[2] = (intensity + color[2]) >> 1;
      }
      else
      {
        overlay_color[0] = overlay_color[1] = overlay_color[2] = intensity;
      }
    }
  }
}

cape_ros::PlanesConstPtr CapeRosNode::generateMessage(const std_msgs::Header& header) const
{
  cape_ros::PlanesPtr planes = boost::make_shared<cape_ros::Planes>();
  for (const PlaneSeg& plane : plane_params_)
  {
    shape_msgs::Plane p;
    p.coef[0] = plane.normal[0];
    p.coef[1] = plane.normal[1];
    p.coef[2] = plane.normal[2];
    p.coef[3] = plane.d;
    planes->planes.push_back(p);

    planes->segments = *cv_bridge::CvImage(header, "mono8", seg_output_).toImageMsg();
  }

  return planes;
}

void CapeRosNode::depthCallback(const sensor_msgs::ImagePtr& image)
{
  if (!intrinsics_ready_)
  {
    try
    {
      sensor_msgs::CameraInfoConstPtr cam_intrinsics_info =
          ros::topic::waitForMessage<sensor_msgs::CameraInfo>("depth/camera_info", ros::Duration(1.0));
      if (cam_intrinsics_info)
      {
        if (cam_intrinsics_info->K[0] > 0 && cam_intrinsics_info->K[4] > 0 && cam_intrinsics_info->K[2] > 0 &&
            cam_intrinsics_info->K[5] > 0)
        {
          fx_ = cam_intrinsics_info->K[0];
          fy_ = cam_intrinsics_info->K[4];
          cx_ = cam_intrinsics_info->K[2];
          cy_ = cam_intrinsics_info->K[5];
          intrinsics_ready_ = true;
        }
      }
      else
      {
        ROS_ERROR("Failed to obtain camera intrinsic messages");
        return;
      }
      ROS_WARN("fx, fy, px, py: %f, %f, %f, %f", fx_, fy_, cx_, cy_);
    }
    catch (std::exception e)
    {
      ROS_ERROR("Failed to obtain camera intrinsic messages: %s", e.what());
      return;
    }
  }

  cv_bridge::CvImageConstPtr image_ptr = cv_bridge::toCvShare(image, image->encoding);
  auto image_depth16 = reinterpret_cast<const uint16_t*>(image_ptr->image.data);
  float scaled_depth(0.);

  if (!cape_extractor_ || height_ != image->height || width_ != image->width)
  {
    height_ = image->height;
    width_ = image->width;
    reset();
  }

  auto iter_mat = depth_mat_.begin<float>();

  // Go through the depth iamge and scale each pixel
  for (int y = 0; y < height_; ++y)
  {
    for (int x = 0; x < width_; ++x)
    {
      scaled_depth = static_cast<float>(*image_depth16) * depth_scale_;

      // Check if the point is valid
      if (scaled_depth <= 0.f)
      {
        *iter_mat = 0.;
      }
      else
        *iter_mat = scaled_depth;

      ++image_depth16;
      ++iter_mat;
    }
  }

  X_ = X_pre_.mul(depth_mat_);
  Y_ = Y_pre_.mul(depth_mat_);
  cloud_array_.setZero();
  seg_output_.setTo(0);
  plane_params_.clear();
  cylinder_params_.clear();

  simpleProjectPointCloud(X_, Y_, depth_mat_, cloud_array_, depth_cutoff_);

  organizePointCloudByCell(cloud_array_, cloud_array_organized_, cell_map_);

  cape_extractor_->process(cloud_array_organized_, seg_output_, plane_params_, cylinder_params_);

  planes_pub_.publish(generateMessage(image->header));

  if (intensity_image_ptr_ && (image_overlay_pub_.getNumSubscribers() > 0))
  {
    overlaySegmentsOnImage(intensity_image_ptr_->image, seg_output_);
    image_overlay_pub_.publish(cv_bridge::CvImage(image->header, "rgb8", overlay_image_).toImageMsg());
  }
}

void CapeRosNode::reset()
{
  cape_extractor_ = std::make_shared<Cape>(height_, width_, patch_size_, patch_size_, cylinder_detection_,
                                           cos_angle_max_, max_merge_dist_);
  depth_mat_.create(height_, width_, CV_32F);
  X_.create(height_, width_, CV_32F);
  Y_.create(height_, width_, CV_32F);
  X_pre_.create(height_, width_, CV_32F);
  Y_pre_.create(height_, width_, CV_32F);

  cloud_array_.resize(width_ * height_, 3);
  cloud_array_organized_.resize(width_ * height_, 3);

  for (auto r = 0; r < height_; ++r)
  {
    for (auto c = 0; c < width_; ++c)
    {
      X_pre_.at<float>(r, c) = static_cast<float>((c - cx_) / fx_);
      Y_pre_.at<float>(r, c) = static_cast<float>((r - cy_) / fy_);
    }
  }

  // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are
  // contiguous
  cell_map_.create(height_, width_, CV_32S);

  const auto nr_horizontal_cells = width_ / patch_size_;
  ///@todo(afakih) join this with the loop above
  for (int r = 0; r < height_; ++r)
  {
    int cell_r = r / patch_size_;
    int local_r = r % patch_size_;
    for (int c = 0; c < width_; ++c)
    {
      int cell_c = c / patch_size_;
      int local_c = c % patch_size_;
      cell_map_.at<int>(r, c) =
          (cell_r * nr_horizontal_cells + cell_c) * patch_size_ * patch_size_ + local_r * patch_size_ + local_c;
    }
  }

  seg_output_.create(height_, width_, CV_8U);
}

void CapeRosNode::intensityCallback(const sensor_msgs::ImagePtr& image)
{
  intensity_image_ptr_ = cv_bridge::toCvShare(image, image->encoding);
}

/**
 * @name  main
 * @brief Initialize ROS node.
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "ros_cape_node");

  ros::NodeHandle nh;
  CapeRosNode node(nh);
  ros::spin();
  return 0;
}

