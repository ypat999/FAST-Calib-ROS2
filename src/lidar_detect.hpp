/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <cstdlib>
#include <filesystem>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_;
    Eigen::Vector3d board_normal_;
    double board_d_;
    std::shared_ptr<rclcpp::Node> node_;

    // 存储中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> filtered_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> plane_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> aligned_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> edge_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> center_z0_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> center_pub_;

    LidarDetect(std::shared_ptr<rclcpp::Node> node, Params &params)
        : node_(node),
          filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          plane_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        board_normal_ = Eigen::Vector3d(params.board_nx, params.board_ny, params.board_nz);
        board_d_ = params.board_d;

        filtered_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("center_cloud", 10);
    }

    void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->clear();
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*filtered_cloud_);
    
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());
        
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Filtered cloud size: %ld", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->clear();
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        if (board_normal_.norm() > 0.5) {
            // Board plane prior: keep a thin band around n·p + d = 0 instead of
            // RANSAC (ground/wall otherwise win when the board is tilted).
            Eigen::Vector3d n = board_normal_.normalized();
            plane_inliers->indices.reserve(filtered_cloud_->size());
            for (size_t i = 0; i < filtered_cloud_->size(); ++i) {
                const pcl::PointXYZ &p = filtered_cloud_->points[i];
                if (std::abs(n.x() * p.x + n.y() * p.y + n.z() * p.z + board_d_) < 0.02)
                    plane_inliers->indices.push_back(i);
            }
            plane_coefficients->values = {n.x(), n.y(), n.z(), board_d_};
            RCLCPP_INFO(node_->get_logger(), "Board band filter: %ld inliers",
                        plane_inliers->indices.size());
        } else {
            pcl::SACSegmentation<pcl::PointXYZ> plane_segmentation;
            plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
            plane_segmentation.setMethodType(pcl::SAC_RANSAC);
            plane_segmentation.setDistanceThreshold(0.01);
            plane_segmentation.setInputCloud(filtered_cloud_);
            plane_segmentation.segment(*plane_inliers, *plane_coefficients);
        }
    
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(node_->get_logger(), "Plane cloud size: %ld", plane_cloud_->size());
    
        // 3. 平面点云对齐
        aligned_cloud_->clear();
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->clear();
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03);
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03);
        boundary_estimator.setAngleThreshold(M_PI / 4);
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        RCLCPP_INFO(node_->get_logger(), "Extracted %ld edge points.", edge_cloud_->size());

        // Debug: dump intermediate clouds when FC_DEBUG_DUMP is set
        if (getenv("FC_DEBUG_DUMP")) {
            std::filesystem::create_directories("/home/ywj/tmp/fc_debug");
            pcl::io::savePCDFileBinary("/home/ywj/tmp/fc_debug/filtered.pcd", *filtered_cloud_);
            pcl::io::savePCDFileBinary("/home/ywj/tmp/fc_debug/plane.pcd", *plane_cloud_);
            pcl::io::savePCDFileBinary("/home/ywj/tmp/fc_debug/aligned.pcd", *aligned_cloud_);
            pcl::io::savePCDFileBinary("/home/ywj/tmp/fc_debug/edge.pcd", *edge_cloud_);
            RCLCPP_INFO(node_->get_logger(), "Plane normal: (%.3f, %.3f, %.3f), d=%.3f, average_z=%.3f",
                        normal.x(), normal.y(), normal.z(), plane_coefficients->values[3], average_z);
        }

        // 5+6. 圆检测。
        // 有板平面先验时（静态 bag 累积点云稀疏、噪声把孔环与板轮廓粘成一团，
        // Euclidean 聚类无法拆出 4 个孔）：直接在全部边缘点上迭代拟合圆并剥离
        // inlier，直到凑够 4 个半径匹配的圆。
        // 无先验时保持原始"聚类 + 逐聚类圆拟合"流程。
        center_z0_cloud_->clear();
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();

        auto accept_circle = [&](const pcl::ModelCoefficients::Ptr &coefficients,
                                 const pcl::PointCloud<pcl::PointXYZ> &cluster,
                                 const pcl::PointIndices::Ptr &inliers) {
            double error = 0.0;
            for (const auto &idx : inliers->indices)
            {
                double dx = cluster.points[idx].x - coefficients->values[0];
                double dy = cluster.points[idx].y - coefficients->values[1];
                error += std::abs(sqrt(dx * dx + dy * dy) - circle_radius_);
            }
            error /= inliers->indices.size();
            if (error >= 0.02)
                return false;

            pcl::PointXYZ center_point;
            center_point.x = coefficients->values[0];
            center_point.y = coefficients->values[1];
            center_point.z = 0.0;
            center_z0_cloud_->push_back(center_point);

            // 将圆心坐标逆变换回原始坐标系
            Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            center_cloud->points.push_back(pcl::PointXYZ(original_point.x(), original_point.y(), original_point.z()));
            return true;
        };

        if (board_normal_.norm() > 0.5) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr work_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            *work_cloud = *edge_cloud_;
            for (int it = 0; it < 8 && center_z0_cloud_->size() < 4 && work_cloud->size() >= 50; ++it) {
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                pcl::SACSegmentation<pcl::PointXYZ> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_CIRCLE2D);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(0.01);
                seg.setMaxIterations(2000);
                seg.setInputCloud(work_cloud);
                seg.segment(*inliers, *coefficients);
                if (inliers->indices.size() < 50)
                    break;
                bool ok = accept_circle(coefficients, *work_cloud, inliers);
                RCLCPP_INFO(node_->get_logger(), "Circle iter %d: c=(%.3f, %.3f) r=%.3f n=%zu %s",
                            it, coefficients->values[0], coefficients->values[1],
                            sqrt(coefficients->values[2]), inliers->indices.size(),
                            ok ? "accepted" : "rejected");
                // 无论接受与否都剥离这批 inlier，避免同一圆被重复拟合
                std::vector<bool> keep(work_cloud->size(), true);
                for (const auto &idx : inliers->indices) keep[idx] = false;
                pcl::PointCloud<pcl::PointXYZ> peeled;
                peeled.reserve(work_cloud->size());
                for (size_t i = 0; i < work_cloud->size(); ++i)
                    if (keep[i]) peeled.push_back(work_cloud->points[i]);
                *work_cloud = peeled;
            }
        } else {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(edge_cloud_);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.02);
            ec.setMinClusterSize(50);
            ec.setMaxClusterSize(20000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(edge_cloud_);
            ec.extract(cluster_indices);

            RCLCPP_INFO(node_->get_logger(), "Number of edge clusters: %ld", cluster_indices.size());

            for (size_t i = 0; i < cluster_indices.size(); ++i)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
                for (const auto &idx : cluster_indices[i].indices) {
                    cluster->push_back(edge_cloud_->points[idx]);
                }

                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                pcl::SACSegmentation<pcl::PointXYZ> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_CIRCLE2D);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(0.01);
                seg.setMaxIterations(1000);
                seg.setInputCloud(cluster);
                seg.segment(*inliers, *coefficients);

                if (inliers->indices.size() > 0)
                    accept_circle(coefficients, *cluster, inliers);
            }
        }
    }

    // 获取中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif