# FAST-Calib (ROS 2)

FAST-Calib is an automatic target-based extrinsic calibration tool for LiDAR-camera systems (eg., [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)).

**Key highlights include:**

1. Support solid-state and mechanical LiDAR.
2. No need for any initial extrinsic parameters.
3. Achieve highly accurate calibration results **in just 2 seconds**.

**In short, it makes extrinsic calibration as simple as intrinsic calibration.**

📬 For further assistance or inquiries, please feel free to contact Chunran Zheng at zhengcr@connect.hku.hk.

<p align="center">
  <img src="./pics/calib.jpg" width="100%">
  <font color=#a0a0a0 size=2>Left: Example of circle extraction from Mid360 point cloud | Right: Point cloud colored with calibrated extrinsic.</font>
</p>

## 1. Prerequisites

PCL>=1.8, OpenCV>=4.0.

## 2. Run our examples

1. Prepare the static acquisition data in `calib_data` folder (see [sample data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/Eq_k_4Mf_11Eggg4a5lbRzgBHwd0EivtCJd2ExtcNlu1FA?e=vjm4gH) from Mid360, Avia and Ouster):

- rosbag containing point cloud messages
- corresponding image

2. Run the calibration process:

```bash
ros2 launch fast_calib calib_launch.py
```

## 3. Run on your own sensor suite

1. Customize the calibration target in the image below.
2. Record data to ros2 bag.
3. if you are using livox ros driver2 make sure that output of your points in PointCloud2 format (edit the launch file of desired livox LiDAR and make sure that xfer_format = 0 )
4. Provide the instrinsic matrix in `qr_params.yaml`.
5. Set distance filter in `qr_params.yaml` for board point cloud (extra points are acceptable).
6. Calibrate now!

### 3.1 Data acquisition tips

- **Keep the sensor rig and the target static while recording.** All point cloud
  frames in a bag are accumulated into a single cloud without motion
  compensation, so a hand-held or moving recording will smear the board and
  break the circle extraction. Mount the rig on a tripod (or put the board on a
  stand) and record 1-2 s per shot.
- Keep the whole board inside both the camera FoV and the LiDAR FoV, roughly
  facing the sensors, and inside the distance filter range (`x_min`~`x_max`,
  `y_min`~`y_max`, `z_min`~`z_max`).
- Capture one image per shot, at the same pose as the bag. Image and bag are
  paired by index in `bag_paths` / `image_paths`.

### 3.2 Multi-shot calibration (recommended)

A single shot already fully constrains the 6-DoF extrinsic (4 circle centers
paired with 4 ArUco centers). Recording several shots from different viewpoints
and letting the tool pick the best consensus is more robust against partial
circle extraction or mis-ordered centers.

1. Record N static shots, each as one bag plus one image.
2. Fill both lists in `qr_params.yaml` (order matters, same index = same shot):

```yaml
bag_paths:
  - "/path/to/shot1_bag"
  - "/path/to/shot2_bag"
  - "/path/to/shot3_bag"
image_paths:
  - "/path/to/shot1.png"
  - "/path/to/shot2.png"
  - "/path/to/shot3.png"
```

3. Launch as usual: `ros2 launch fast_calib calib_launch.py`

How it works:

1. Each shot is solved independently (detection + SVD), producing its own
   extrinsic and RMSE. Shots with incomplete detections are skipped.
2. An adaptive inlier threshold is derived from the median single-shot RMSE
   (`max(3 x median, 0.04 m)`).
3. A RANSAC-style consensus keeps the shots that agree with the best-supported
   extrinsic; inconsistent shots (bad detection, wrong center ordering) are
   reported as outliers and discarded.
4. All inlier correspondences are stacked and solved once with a joint SVD,
   giving the least-squares optimal extrinsic over all shots.

The terminal output lists per-shot `single_rmse` / `final_rmse` with an
`INLIER` / `outlier` flag, plus the global RMSE and the final transformation.
If `bag_paths` / `image_paths` are left empty, the tool falls back to the
single `bag_path` / `image_path`.

<p align="center">
  <img src="./pics/calibration_target.jpg" width="100%">
  <font color=#a0a0a0 size=2>Left: Actual calibration target | Right: Technical drawing with annotated dimensions.</font>
</p>

## 4. Appendix

Related article is coming soon...

The calibration target design is based on the [velo2cam_calibration](https://github.com/beltransen/velo2cam_calibration).

For further details on the algorithm workflow, see [this document](https://github.com/xuankuzcr/FAST-Calib/blob/main/workflow.md).

## 5. Acknowledgments

Special thanks to [Jiaming Xu](https://github.com/Xujiaming1) for his support, [Haotian Li](https://github.com/luo-xue) for the equipment, and the [velo2cam_calibration](https://github.com/beltransen/velo2cam_calibration) algorithm.
