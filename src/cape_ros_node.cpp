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

CapeRosNode::CapeRosNode(ros::NodeHandle nh)
      : nh_(nh)
      , cos_angle_max_(std::cos(M_PI/30))
      , max_merge_dist_(50.0)
      , depth_cutoff_(5.0)
      , cylinder_detection_(false)
      , patch_size_(20)
    {
        depth_sub = nh.subscribe("depth_in", 2, &CapeRosNode::depthCallback, this);
        intensity_sub = nh.subscribe("ir_in", 2, &CapeRosNode::intensityCallback, this);
        planes_pub_ = nh.advertise<cape_ros::Planes>("planes", 1);
    }

    void CapeRosNode::simpleProjectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, Eigen::MatrixXf& cloud_array, const double& depth_cutoff) {
      int width = X.cols;
      int height = X.rows;

      float *sz, *sx, *sy;
      int id;
      for (int r = 0; r < height; r++) {
        sx = X.ptr<float>(r);
        sy = Y.ptr<float>(r);
        sz = Z.ptr<float>(r);
        for (int c = 0; c < width; c++) {
          if (sz[c] < depth_cutoff) {
            id = r * width + c;
            cloud_array(id, 0) = sx[c];
            cloud_array(id, 1) = sy[c];
            cloud_array(id, 2) = sz[c];
          }
        }
      }
    }

    void CapeRosNode::organizePointCloudByCell(Eigen::MatrixXf& cloud_in, Eigen::MatrixXf& cloud_out, cv::Mat& cell_map) {
      int width = cell_map.cols;
      int height = cell_map.rows;
      int mxn = width * height;
      int mxn2 = 2 * mxn;

      int id, it(0);
      int* cell_map_ptr;
      for (int r = 0; r < height; r++) {
        cell_map_ptr = cell_map.ptr<int>(r);
        for (int c = 0; c < width; c++) {
          id = cell_map_ptr[c];
          *(cloud_out.data() + id) = *(cloud_in.data() + it);
          *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
          *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
          it++;
        }
      }
    }

    void CapeRosNode::drawResult(const cv::Mat& img, const cv::Mat& seg, const double& time_elapsed) {
      // Populate with random color codes
      std::vector<cv::Vec3b> color_code;
      for (int i = 0; i < 100; i++) {
        cv::Vec3b color;
        color[0] = rand() % 255;
        color[1] = rand() % 255;
        color[2] = rand() % 255;
        color_code.push_back(color);
      }

      // Add specific colors for planes
      color_code[0][0] = 0;
      color_code[0][1] = 0;
      color_code[0][2] = 255;
      color_code[1][0] = 255;
      color_code[1][1] = 0;
      color_code[1][2] = 204;
      color_code[2][0] = 255;
      color_code[2][1] = 100;
      color_code[2][2] = 0;
      color_code[3][0] = 0;
      color_code[3][1] = 153;
      color_code[3][2] = 255;
      // Add specific colors for cylinders
      color_code[50][0] = 178;
      color_code[50][1] = 255;
      color_code[50][2] = 0;
      color_code[51][0] = 255;
      color_code[51][1] = 0;
      color_code[51][2] = 51;
      color_code[52][0] = 0;
      color_code[52][1] = 255;
      color_code[52][2] = 51;
      color_code[53][0] = 153;
      color_code[53][1] = 0;
      color_code[53][2] = 255;

      int width = img.cols;
      int height = img.rows;

      cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(img.rows, img.cols, cv::Vec3b(0, 0, 0));
      // Map segments with color codes and overlap segmented image w/ RGB
      uchar* dColor;
      int code;
      for (int r = 0; r < height; r++) {
        dColor = seg_rz.ptr<uchar>(r);
        const auto * sCode = seg.ptr<uchar>(r);
        const uchar* sir = img.ptr<uchar>(r);
        for (int c = 0; c < width; c++) {
          code = *sCode;
          if (code > 0) {
            dColor[c * 3] = color_code[code - 1][0] / 2 + sir[0] / 2;
            dColor[c * 3 + 1] = color_code[code - 1][1] / 2 + sir[0] / 2;
            dColor[c * 3 + 2] = color_code[code - 1][2] / 2 + sir[0] / 2;
          } else {
            dColor[c * 3] = sir[0];
            dColor[c * 3 + 1] = sir[0];
            dColor[c * 3 + 2] = sir[0];
          }
          sCode++;
          sir++;
        }
      }

      // Show frame rate and labels
      cv::rectangle(seg_rz, cv::Point(0, 0), cv::Point(width, 20), cv::Scalar(0, 0, 0), -1);
      std::stringstream fps;
      fps << (int)(1 / time_elapsed + 0.5) << " fps";
      cv::putText(seg_rz, fps.str(), cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

      cv::imshow("plane_extractor", seg_rz);
      cv::waitKey(15);
    }

    cape_ros::PlanesConstPtr CapeRosNode::generateMessage() const
    {
       cape_ros::PlanesPtr planes = boost::make_shared<cape_ros::Planes>();
       for (const PlaneSeg& plane : plane_params)
       {
         shape_msgs::Plane p;
         p.coef[0] = plane.normal[0];
         p.coef[1] = plane.normal[1];
         p.coef[2] = plane.normal[2];
         p.coef[3] = plane.d;
         planes->planes.push_back(p);
         cv_bridge::CvImage Im(std_msgs::Header(), "mono8", seg_output);
         planes->segments = *Im.toImageMsg();
       }

       return planes;

    }

    void CapeRosNode::depthCallback(const sensor_msgs::ImagePtr &image) {
      ROS_DEBUG("DepthCallback");
      if (!intrinsics_ready) {
        try {
          sensor_msgs::CameraInfoConstPtr cam_intrinsics_info =
              ros::topic::waitForMessage<sensor_msgs::CameraInfo>("depth/camera_info", ros::Duration(1.0));
          if (cam_intrinsics_info) {
            if (cam_intrinsics_info->K[0] > 0 && cam_intrinsics_info->K[4] > 0 && cam_intrinsics_info->K[2] > 0 && cam_intrinsics_info->K[5] > 0) {
              fx = cam_intrinsics_info->K[0];
              fy = cam_intrinsics_info->K[4];
              cx = cam_intrinsics_info->K[2];
              cy = cam_intrinsics_info->K[5];
              intrinsics_ready = true;
            }
          } else {
            ROS_ERROR("Failed to obtain camera intrinsic messages");
            return;
          }
          ROS_WARN("fx, fy, px, py: %f, %f, %f, %f", fx, fy, cx, cy);
        } catch (std::exception e) {
          ROS_ERROR("Failed to obtain camera intrinsic messages: %s", e.what());
          return;
        }
      }

      cv_bridge::CvImageConstPtr image_ptr = cv_bridge::toCvShare(image, image->encoding);
      auto image_depth16 = reinterpret_cast<const uint16_t *>(image_ptr->image.data);
      float scaled_depth(0.);

      if (!cape_extractor_ || height_ != image->height || width_ != image->width)
      {
        height_ = image->height;
        width_ =  image->width;
        reset();
      }


      depth_mat.create(height_, width_, CV_32F);
      auto iter_mat = depth_mat.begin<float>();

      // Go through the depth iamge and scale each pixel
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          scaled_depth = static_cast<float>(*image_depth16) * DEPTH_SCALE_METERS;

          // Check if the point is valid
          if (scaled_depth <= 0.f) {
            *iter_mat = 0.;
          } else
            *iter_mat = scaled_depth;

          ++image_depth16;
          ++iter_mat;
        }
      }

      ROS_WARN_STREAM("DONE MAPPING DEPTH");

      // cv::imshow("depth", depth_img);
      // cv::waitKey(10);

      // Backproject to point cloud
      X = X_pre.mul(depth_mat);
      Y = Y_pre.mul(depth_mat);
      cloud_array.setZero();

      // The following transformation+projection is only necessary to visualize RGB with overlapped segments
      // Transform point cloud to color reference frame
      // X_t = ((float)R_stereo.at<double>(0,0))*X+((float)R_stereo.at<double>(0,1))*Y+((float)R_stereo.at<double>(0,2))*d_img +
      // (float)t_stereo.at<double>(0);
      // Y_t = ((float)R_stereo.at<double>(1,0))*X+((float)R_stereo.at<double>(1,1))*Y+((float)R_stereo.at<double>(1,2))*d_img +
      // (float)t_stereo.at<double>(1);
      // d_img = ((float)R_stereo.at<double>(2,0))*X+((float)R_stereo.at<double>(2,1))*Y+((float)R_stereo.at<double>(2,2))*d_img +
      // (float)t_stereo.at<double>(2);
      // projectPointCloud(X_t, Y_t, d_img, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb, t_stereo.at<double>(2), cloud_array);

      simpleProjectPointCloud(X, Y, depth_mat, cloud_array, depth_cutoff_);
      ROS_WARN_STREAM("Simple Projected");

      seg_rz.setTo(0);
      seg_output.setTo(0);

      // Run CAPE
      plane_params.clear();
      cylinder_params.clear();

      double t1 = cv::getTickCount();
      organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);

      cape_extractor_->process(cloud_array_organized, seg_output, plane_params, cylinder_params);

      double t2 = cv::getTickCount();
      double time_elapsed = (t2 - t1) / (double)cv::getTickFrequency();
      cout << "CAPE initial time elapsed: " << time_elapsed << endl;

      // t1 = cv::getTickCount();
      //// hough lines on seg
      // cv::Mat_<uchar> seg_binary = cv::Mat_<uchar>(height, width, uchar(0));
      // cv::threshold(seg_output, seg_binary, 0, 255, 0);
      // cv::Mat_<uchar> seg_edge;
      // cv::Canny(seg_binary, seg_edge, 50, 200, 3);

      // std::vector<cv::Vec4i> lines;
      // cv::Mat seg_edge_lines;
      // cv::cvtColor(seg_edge, seg_edge_lines, CV_GRAY2BGR);
      //// cv::HoughLinesP(seg_edge, lines, 1, CV_PI/180, 50, 50, 10);
      // cv::HoughLinesP(seg_edge, lines, 1, CV_PI / 180, 50, 50, 50);

      // t2 = cv::getTickCount();
      // time_elapsed = (t2 - t1) / (double)cv::getTickFrequency();
      // cout << "Canny+HoughLinesP time elapsed: " << time_elapsed << endl;

      // for (size_t i = 0; i < lines.size(); i++) {
      //  cv::Vec4i l = lines[i];
      //  cv::line(seg_edge_lines, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, CV_AA);
      //}

      // cv::imshow("seg_edge", seg_edge);
      // cv::imshow("seg_binary", seg_binary);
      // cv::imshow("detected lines", seg_edge_lines);
      // cv::waitKey(10);

      /* Uncomment this block to print model params */
      for (int p_id = 0; p_id < plane_params.size(); p_id++) {
        cout << "[Plane #" << p_id << "] with ";
        cout << "normal: (" << plane_params[p_id].normal[0] << " " << plane_params[p_id].normal[1] << " " << plane_params[p_id].normal[2] << "), ";
        cout << "mean: (" << plane_params[p_id].mean[0] << " " << plane_params[p_id].mean[1] << " " << plane_params[p_id].mean[2] << "), ";
        cout << "score: " << plane_params[p_id].score << ", ";
        cout << "d: " << plane_params[p_id].d << endl;
      }

      for (int c_id = 0; c_id < cylinder_params.size(); c_id++) {
        cout << "[Cylinder #" << c_id << "] with ";
        cout << "axis: (" << cylinder_params[c_id].axis[0] << " " << cylinder_params[c_id].axis[1] << " " << cylinder_params[c_id].axis[2] << "), ";
        cout << "center: (" << cylinder_params[c_id].centers[0].transpose() << "), ";
        cout << "radius: " << cylinder_params[c_id].radii[0] << endl;
      }

      planes_pub_.publish(generateMessage());

      //planes_.plane_mask = seg_output.clone();
      //planes_.plane_array.clear();
      /*for (std::size_t i = 0; i < plane_params.size(); ++i) {
        const PlaneSeg& pl = plane_params[i];
        Plane plane(i + 1, pl.mean[0], pl.mean[1], pl.mean[2], pl.normal[0], pl.normal[1], pl.normal[2]);
        planes_.plane_array.push_back(plane);
      }*/
      std::cout << "Done all " << std::endl;
      if (intensity_image_ptr_)
      {
        drawResult(intensity_image_ptr_->image, seg_output, time_elapsed);
      }
      else
{
  std::cout << " Intensity image not initialized " << std::endl;
}

    }

    void CapeRosNode::reset()
    {
        ROS_WARN_STREAM("RESETTING");
        nr_horizontal_cells = width_ / patch_size_;
        nr_vertical_cells = height_ / patch_size_;

        cape_extractor_ = std::make_shared<Cape>(height_, width_, patch_size_, patch_size_, cylinder_detection_, cos_angle_max_, max_merge_dist_);
        X.create(height_, width_, CV_32F);
        Y.create(height_, width_, CV_32F);
        X_pre.create(height_, width_, CV_32F);
        Y_pre.create(height_, width_, CV_32F);
        //U.create(height_, width_, CV_32F);
        //V.create(height_, width_, CV_32F);

        cloud_array.resize(width_ * height_, 3);
        cloud_array_organized.resize(width_ * height_, 3);

        for (int r = 0; r < height_; r++) {
          for (int c = 0; c < width_; c++) {
            X_pre.at<float>(r, c) = (c - cx) / fx;
            Y_pre.at<float>(r, c) = (r - cy) / fy;
          }
        }

        // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
        cell_map.create(height_, width_, CV_32S);

        for (int r = 0; r < height_; r++) {
          int cell_r = r / patch_size_;
          int local_r = r % patch_size_;
          for (int c = 0; c < width_; c++) {
            int cell_c = c / patch_size_;
            int local_c = c % patch_size_;
            cell_map.at<int>(r, c) = (cell_r * nr_horizontal_cells + cell_c) * patch_size_ * patch_size_ + local_r * patch_size_ + local_c;
          }
        }

        seg_rz.create(height_, width_, CV_8UC3);
        seg_output.create(height_, width_, CV_8U);
        ROS_WARN_STREAM("RESETTED");
    }

    void CapeRosNode::intensityCallback(const sensor_msgs::ImagePtr &image) {
      ROS_DEBUG("IRCallback");
      intensity_image_ptr_ = cv_bridge::toCvShare(image, image->encoding);
    }


/**
 * @name  main
 * @brief Initialize ROS node.
 */
int main(int argc, char **argv) {
  ros::init(argc, argv, "ros_cape_node");

  ros::NodeHandle nh;
  CapeRosNode node(nh);
  ros::spin();
  return 0;
}
