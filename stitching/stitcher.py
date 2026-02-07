#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os

def preprocess_rgbd_pair(i, rgb_dir, depth_dir):
    """
    Preprocesses a pair of RGB and depth images to create an Open3D RGBD image.

    Args:
        i (int): The index of the image pair to process.
        rgb_dir (str): Path to the directory containing RGB images.
        depth_dir (str): Path to the directory containing depth images.

    Returns:
        o3d.geometry.RGBDImage: An Open3D RGBD image created from the specified RGB and depth images.
    """
    im_depth = o3d.io.read_image(os.path.join(depth_dir, f"depth_fixed_{i:03d}.png"))
    im_color = o3d.io.read_image(os.path.join(rgb_dir, f"rgb_{i:03d}.png"))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_color, im_depth,
        depth_scale=1000.0,
        depth_trunc=8.0,
        convert_rgb_to_intensity=False
    )
    return rgbd

def preprocess_pointcloud(rgbd, intrinsic, extrinsic, voxel_size):
    """
    Preprocesses an RGBD image to generate a point cloud and its downsampled version.

    Args:
        rgbd (o3d.geometry.RGBDImage): The RGBD image to process.
        intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters of the camera.
        extrinsic (numpy.ndarray): The extrinsic transformation matrix for the camera.
        voxel_size (float): The size of the voxel for downsampling the point cloud.

    Returns:
        tuple:
            - o3d.geometry.PointCloud: The original point cloud generated from the RGBD image.
            - o3d.geometry.PointCloud: The downsampled point cloud with estimated normals.
    """
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    return pcd, pcd_down

def run_colored_icp(source_down, target_down, init_transform, max_corr_dist):
    """
    Runs the Colored Iterative Closest Point (ICP) algorithm to align two point clouds.

    Args:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        init_transform (numpy.ndarray): The initial transformation matrix to align the source to the target.
        max_corr_dist (float): The maximum correspondence distance for point cloud alignment.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the Colored ICP algorithm, containing the transformation matrix and fitness score.
    """
    return o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, max_corr_dist, init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )

def main():
    parser = argparse.ArgumentParser(description="Colored ICP-based point cloud stitching")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory JSON file")
    parser.add_argument("--rgb_dir", required=True, help="Directory containing RGB images")
    parser.add_argument("--depth_dir", required=True, help="Directory containing depth images")
    parser.add_argument("--nframes", type=int, required = True, help="Number of frames to process (default: 20)")
    args = parser.parse_args()

    trajectory = o3d.io.read_pinhole_camera_trajectory(args.trajectory)
    intrinsic  = trajectory.parameters[0].intrinsic

    if args.nframes < len(trajectory.parameters):
        nframes = args.nframes
    else:
        nframes = len(trajectory.parameters)
    print(f"Using {nframes} frames for stitching")

    voxel_down    = 0.01
    icp_corr_dist = 0.05

    accumulated_pcd = o3d.geometry.PointCloud()
    T_global        = np.identity(4)

    rgbd_0 = preprocess_rgbd_pair(0, args.rgb_dir, args.depth_dir)
    extr_0 = trajectory.parameters[0].extrinsic
    pcd_0, pcd_0_down = preprocess_pointcloud(rgbd_0, intrinsic, extr_0, voxel_down)
    accumulated_pcd += pcd_0

    for i in range(0, nframes):
        print(f"Processing frame {i} -> {i+1}...")

        rgbd_i = preprocess_rgbd_pair(i, args.rgb_dir, args.depth_dir)
        rgbd_j = preprocess_rgbd_pair(i + 1, args.rgb_dir, args.depth_dir)

        extr_i = trajectory.parameters[i].extrinsic
        extr_j = trajectory.parameters[i + 1].extrinsic

        pcd_i, pcd_i_down = preprocess_pointcloud(rgbd_i, intrinsic, extr_i, voxel_down)
        pcd_j, pcd_j_down = preprocess_pointcloud(rgbd_j, intrinsic, extr_j, voxel_down)

        src_cent = np.asarray(pcd_i_down.get_center())
        tgt_cent = np.asarray(pcd_j_down.get_center())
        print(f"Frame {i}->{i+1} | Centroid distance: {np.linalg.norm(src_cent - tgt_cent):.3f}")

        init_transform = extr_j @ np.linalg.inv(extr_i)

        print(f"pcd_i_down has {len(pcd_i_down.points)} points, pcd_j_down has {len(pcd_j_down.points)} points")
        src_cent = np.asarray(pcd_i_down.get_center())
        tgt_cent = np.asarray(pcd_j_down.get_center())
        src_cent_in_j = (init_transform @ np.hstack((src_cent, 1)))[:3]

        print("raw centroids:", src_cent, tgt_cent)
        print("transformed src -> j:", src_cent_in_j, "distance:",
              np.linalg.norm(src_cent_in_j - tgt_cent))

        icp_result = run_colored_icp(pcd_i_down, pcd_j_down, init_transform, icp_corr_dist)

        T_global = T_global @ np.linalg.inv(icp_result.transformation)
        pcd_j_global = pcd_j.transform(T_global)
        accumulated_pcd += pcd_j_global

        bbox = accumulated_pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        density = len(accumulated_pcd.points) / volume
        print(f"Density: {density:.2f} points per unit volume")

    print('here')
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.03)

    accumulated_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    accumulated_pcd.orient_normals_consistent_tangent_plane(100)
    accumulated_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # o3d.visualization.draw_geometries([accumulated_pcd], point_show_normal=False)

    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(accumulated_pcd, depth=11)
    densities = np.asarray(densities)
    vertices_to_keep = densities > np.percentile(densities, 5)
    mesh_poisson = mesh_poisson.select_by_index(np.where(vertices_to_keep)[0])
    bbox = accumulated_pcd.get_axis_aligned_bounding_box()
    mesh_poisson = mesh_poisson.crop(bbox)

    mesh_poisson.compute_vertex_normals()
    z_vals = np.asarray(mesh_poisson.vertices)[:, 2]
    norm = plt.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    cmap = cm.get_cmap("terrain")
    colors_mapped = cmap(norm(z_vals))[:, :3]
    mesh_poisson.vertex_colors = o3d.utility.Vector3dVector(colors_mapped)
    
    crater_name = os.path.basename(os.path.normpath(args.depth_dir))
    mesh_filename = f"stitched_mesh_{crater_name}.ply"
    o3d.io.write_triangle_mesh(mesh_filename, mesh_poisson)

    o3d.visualization.draw_geometries(
        [mesh_poisson], window_name="Height-colored Crater", mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()