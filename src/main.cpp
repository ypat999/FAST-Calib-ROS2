/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <limits>

namespace {

struct CalibShot {
  std::string bag_path;
  std::string image_path;
};

struct ShotResult {
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_centers;  // 排序后的 LiDAR 圆心
  pcl::PointCloud<pcl::PointXYZ>::Ptr qr_centers;     // 排序后的相机圆心
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud;      // 原始融合点云（用于可视化）
  cv::Mat annotated_image;                            // 该角度的检测可视化图
  Eigen::Matrix4f T;                                  // 单角度独立解
  double rmse;                                        // 单角度 RMSE
  bool valid;
};

// 构建 shots 列表：优先 bag_paths/image_paths，否则回退到单个 bag_path/image_path
std::vector<CalibShot> buildShots(const Params& params) {
  std::vector<CalibShot> shots;
  if (!params.bag_paths.empty() || !params.image_paths.empty()) {
    size_t n = std::max(params.bag_paths.size(), params.image_paths.size());
    shots.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      CalibShot s;
      s.bag_path = i < params.bag_paths.size() ? params.bag_paths[i] : params.bag_paths.back();
      s.image_path = i < params.image_paths.size() ? params.image_paths[i] : params.image_paths.back();
      shots.push_back(s);
    }
  } else {
    shots.push_back({params.bag_path, params.image_path});
  }
  return shots;
}

// 用 T 把 LiDAR 圆心变换到相机系，返回与相机圆心的对应点 RMSE
double shotRMSE(const pcl::PointCloud<pcl::PointXYZ>::Ptr& lidar,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr& qr,
                const Eigen::Matrix4f& T) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  alignPointCloud(lidar, aligned, T);
  return computeRMSE(qr, aligned);
}

}  // namespace

int main(int argc, char **argv) 
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mono_qr_pattern");

    // Load parameters
    Params params = loadParameters(node);

    // Initialize QR detection and LiDAR detection (reused across shots)
    QRDetectPtr qrDetectPtr(new QRDetect(node, params));
    LidarDetectPtr lidarDetectPtr(new LidarDetect(node, params));

    std::vector<CalibShot> shots = buildShots(params);
    if (shots.empty()) {
      RCLCPP_ERROR(node->get_logger(), "No calibration shots configured.");
      rclcpp::shutdown();
      return 1;
    }
    RCLCPP_INFO(node->get_logger(), "Calibrating %zu shot(s)...", shots.size());

    // --- Pass 1: detect and solve each shot independently ---
    std::vector<ShotResult> results;
    results.reserve(shots.size());

    for (size_t s = 0; s < shots.size(); ++s) {
      ShotResult r;
      r.lidar_centers.reset(new pcl::PointCloud<pcl::PointXYZ>);
      r.qr_centers.reset(new pcl::PointCloud<pcl::PointXYZ>);
      r.raw_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
      r.T = Eigen::Matrix4f::Identity();
      r.rmse = -1.0;
      r.valid = false;

      RCLCPP_INFO(node->get_logger(), "[Shot %zu/%zu] bag=%s image=%s",
                  s + 1, shots.size(), shots[s].bag_path.c_str(), shots[s].image_path.c_str());

      DataPreprocess preproc(params, shots[s].bag_path, shots[s].image_path);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input = preproc.cloud_input_;
      cv::Mat img_input = preproc.img_input_;

      if (img_input.empty() || !cloud_input || cloud_input->empty()) {
        RCLCPP_WARN(node->get_logger(), "[Shot %zu] Empty image or point cloud, skip.", s + 1);
        results.push_back(r);
        continue;
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr qr_center_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_center_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      qrDetectPtr->detect_qr(img_input, qr_center_cloud);
      lidarDetectPtr->detect_lidar(cloud_input, lidar_center_cloud);

      // 保留该角度的检测可视化图（imageCopy_ 会被后续角度覆盖）
      r.annotated_image = qrDetectPtr->imageCopy_.empty() ? img_input.clone()
                                                          : qrDetectPtr->imageCopy_.clone();

      if (qr_center_cloud->size() != TARGET_NUM_CIRCLES ||
          lidar_center_cloud->size() != TARGET_NUM_CIRCLES) {
        RCLCPP_WARN(node->get_logger(),
                    "[Shot %zu] Detected %zu QR / %zu LiDAR centers (expected %d), skip.",
                    s + 1, qr_center_cloud->size(), lidar_center_cloud->size(), TARGET_NUM_CIRCLES);
        results.push_back(r);
        continue;
      }

      sortPatternCenters(qr_center_cloud, r.qr_centers, "camera");
      sortPatternCenters(lidar_center_cloud, r.lidar_centers, "lidar");
      if (r.qr_centers->size() != TARGET_NUM_CIRCLES ||
          r.lidar_centers->size() != TARGET_NUM_CIRCLES) {
        RCLCPP_WARN(node->get_logger(), "[Shot %zu] Failed to sort centers, skip.", s + 1);
        results.push_back(r);
        continue;
      }

      pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
      svd.estimateRigidTransformation(*r.lidar_centers, *r.qr_centers, r.T);
      r.rmse = shotRMSE(r.lidar_centers, r.qr_centers, r.T);
      r.raw_cloud = cloud_input;
      r.valid = true;

      RCLCPP_INFO(node->get_logger(), "[Shot %zu] Single-shot RMSE = %.4f m", s + 1, r.rmse);
      results.push_back(r);
    }

    // Collect valid shots
    std::vector<int> valid_idx;
    for (size_t i = 0; i < results.size(); ++i)
      if (results[i].valid) valid_idx.push_back(static_cast<int>(i));

    if (valid_idx.empty()) {
      RCLCPP_ERROR(node->get_logger(), "No valid shot found, cannot calibrate.");
      rclcpp::shutdown();
      return 1;
    }

    // --- Pass 2: adaptive inlier threshold from median single-shot RMSE ---
    std::vector<double> rmses;
    for (int i : valid_idx) rmses.push_back(results[i].rmse);
    std::sort(rmses.begin(), rmses.end());
    double med = rmses[rmses.size() / 2];
    double inlier_thr = std::max(med * 3.0, 0.04);
    RCLCPP_INFO(node->get_logger(), "Median single-shot RMSE = %.4f m, inlier threshold = %.4f m",
                med, inlier_thr);

    // --- Pass 3: RANSAC-style consensus (vote for the extrinsic that explains most shots) ---
    int best = valid_idx[0];
    size_t best_inliers = 0;
    double best_cost = std::numeric_limits<double>::max();
    for (int i : valid_idx) {
      size_t inl = 0;
      double cost = 0.0;
      for (int j : valid_idx) {
        double e = shotRMSE(results[j].lidar_centers, results[j].qr_centers, results[i].T);
        if (e < inlier_thr) { ++inl; cost += e; }
      }
      if (inl > best_inliers || (inl == best_inliers && cost < best_cost)) {
        best = i; best_inliers = inl; best_cost = cost;
      }
    }

    // Build the inlier set w.r.t. the winning extrinsic
    std::vector<int> inlier_idx;
    for (int j : valid_idx) {
      if (shotRMSE(results[j].lidar_centers, results[j].qr_centers, results[best].T) < inlier_thr)
        inlier_idx.push_back(j);
    }
    RCLCPP_INFO(node->get_logger(), "Consensus: %zu / %zu valid shots used.",
                inlier_idx.size(), valid_idx.size());

    // --- Pass 4: joint refinement (stack all inlier correspondences, solve once) ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr all_lidar(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr all_qr(new pcl::PointCloud<pcl::PointXYZ>);
    for (int j : inlier_idx) {
      *all_lidar += *results[j].lidar_centers;
      *all_qr += *results[j].qr_centers;
    }

    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    Eigen::Matrix4f transformation;
    svd.estimateRigidTransformation(*all_lidar, *all_qr, transformation);
    double global_rmse = shotRMSE(all_lidar, all_qr, transformation);

    // Per-shot residual under the final extrinsic
    std::cout << BOLDYELLOW << "[Result] Multi-shot calibration summary:" << RESET << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
      bool used = std::find(inlier_idx.begin(), inlier_idx.end(), static_cast<int>(i)) != inlier_idx.end();
      std::cout << "  [shot " << (i + 1) << "] "
                << (results[i].valid ? "valid  " : "invalid")
                << (used ? " INLIER" : " outlier");
      if (results[i].valid) {
        double final_e = shotRMSE(results[i].lidar_centers, results[i].qr_centers, transformation);
        std::cout << " | single_rmse=" << std::fixed << std::setprecision(4) << results[i].rmse
                  << " m | final_rmse=" << final_e << " m";
      }
      std::cout << std::endl;
    }

    std::cout << BOLDYELLOW << "[Result] Global RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
              << global_rmse << " m" << RESET << std::endl;
    std::cout << BOLDYELLOW << "[Result] Extrinsic parameters T_cam_lidar: " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

    // Visualize using the first inlier shot
    int viz_idx = inlier_idx[0];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(results[viz_idx].raw_cloud, transformation, qrDetectPtr->cameraMatrix_,
                             qrDetectPtr->distCoeffs_, results[viz_idx].annotated_image, colored_cloud);

    saveCalibrationResults(params, transformation, colored_cloud, results[viz_idx].annotated_image);

    // Publish the colored cloud for RViz inspection
    auto colored_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", 1);
    auto aligned_lidar_centers_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_lidar_centers", 1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_centers(new pcl::PointCloud<pcl::PointXYZ>);
    alignPointCloud(results[viz_idx].lidar_centers, aligned_centers, transformation);

    rclcpp::Rate rate(1);
    while (rclcpp::ok()) {
      sensor_msgs::msg::PointCloud2 colored_cloud_msg;
      pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
      colored_cloud_msg.header.stamp = node->get_clock()->now();
      colored_cloud_msg.header.frame_id = "map";
      colored_cloud_pub->publish(colored_cloud_msg);

      if (DEBUG) {
        sensor_msgs::msg::PointCloud2 qr_centers_msg;
        pcl::toROSMsg(*results[viz_idx].qr_centers, qr_centers_msg);
        qr_centers_msg.header = colored_cloud_msg.header;
        qrDetectPtr->qr_pub_->publish(qr_centers_msg);

        sensor_msgs::msg::PointCloud2 lidar_centers_msg;
        pcl::toROSMsg(*results[viz_idx].lidar_centers, lidar_centers_msg);
        lidar_centers_msg.header = colored_cloud_msg.header;
        lidarDetectPtr->center_pub_->publish(lidar_centers_msg);

        sensor_msgs::msg::PointCloud2 aligned_lidar_centers_msg;
        pcl::toROSMsg(*aligned_centers, aligned_lidar_centers_msg);
        aligned_lidar_centers_msg.header = colored_cloud_msg.header;
        aligned_lidar_centers_pub->publish(aligned_lidar_centers_msg);
      }

      rclcpp::spin_some(node);
      rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
