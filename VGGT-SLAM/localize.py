import argparse
import os
import time
import glob
import cv2
import numpy as np
import torch
import viser
import viser.transforms as viser_tf
from termcolor import colored
from scipy.spatial.transform import Rotation
import scipy.stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

from vggt.vggt.models.vggt import VGGT
from vggt_slam.solver import Solver
from vggt_slam.h_solve import ransac_projective, apply_homography
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

parser = argparse.ArgumentParser(description="VGGT-SLAM Multi-Image Localization")
parser.add_argument("--map_path", type=str, default="vggt_slam_map.pth", help="Path to the pre-built map state file (.pth)")
parser.add_argument("--localization_dir", type=str, default="localization_config.txt", help="")

# 添加 Viewer 类的定义
class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        
        # Global toggle for all frames and frustums
        self.gui_show_frames = self.server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        self.gui_show_frames.on_update(self._on_update_show_frames)
        
        # Store frames and frustums by submap
        self.submap_frames: dict[int, list[viser.FrameHandle]] = {}
        self.submap_frustums: dict[int, list[viser.CameraFrustumHandle]] = {}
        
        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
    
    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()
        
        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []
        
        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)
            
            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible


def decompose_pose_from_vggt(H_new, vggt_intrinsic, vggt_extrinsic_inv):
    """
    Decomposes the final SL(4) global pose H_new into rotation and translation components.

    Args:
        H_new (np.ndarray): Final SL(4) global pose (4x4).
        vggt_intrinsic (np.ndarray): VGGT intrinsic matrix for the new image (3x3).
        vggt_extrinsic_inv (np.ndarray): VGGT inverse extrinsic matrix for the new image (4x4, T_cam_ref).
                                        In our case, it is the identity matrix.

    Returns:
        tuple: (R, t)
            R (np.ndarray): 3x3 rotation matrix.
            t (np.ndarray): 3x1 translation vector.
    """
    # 1. P_proj = K * [R|t]_cam * H_inv_world
    H_inv_new = np.linalg.inv(H_new)
    T_ref_cam_3x4 = vggt_extrinsic_inv[:3, :]

    projection_matrix = vggt_intrinsic @ T_ref_cam_3x4 @ H_inv_new

    _, R_mat, t_vec = cv2.decomposeProjectionMatrix(projection_matrix)[:3]
    t_vec = t_vec / t_vec[3, 0]
    
    R_se3 = np.linalg.inv(R_mat)
    t_se3 = t_vec[:3, 0]

    return R_se3, t_se3


def localize_image(solver, model, new_image_path):
    """
    localize a new image against the pre-built map using VGGT.

    Args:
        solver (Solver): instance of the Solver class containing the map and retrieval model.
        model (VGGT): loaded VGGT model.
        new_image_path (str): path to the new image to be localized.

    Returns:
        np.ndarray or None: computed global pose matrix (H_new) if successful, otherwise None.
    """
    device = solver.device

    # Step 1: Find the best matching keyframe in the map
    new_image_cv = cv2.imread(new_image_path)
    if new_image_cv is None:
        print(f"Error: Could not read image at {new_image_path}")
        return None
    
    # Convert BGR to RGB for correct color processing
    new_image_cv = cv2.cvtColor(new_image_cv, cv2.COLOR_BGR2RGB)
    
    # 1.1 get the feature query vector for the new image
    query_vector = solver.image_retrieval.get_single_embeding(new_image_cv)

    # 1.2 search for the best match in the map
    # set the current submap ID to -1 and do not ignore any submaps
    best_score, best_submap_id, best_frame_index = solver.map.retrieve_best_score_frame(query_vector, -1, False)

    if best_submap_id is None:
        print(colored(f"Localization failed for {os.path.basename(new_image_path)}: No suitable match found in the map.", "red"))
        return None

    best_match_submap = solver.map.get_submap(best_submap_id)

    # Step 2: Run VGGT on the image pair to get relative geometry
    new_image_frame_tensor = load_and_preprocess_images([new_image_path]).to(device)

    # get the tensor for the best match frame
    best_match_frame = best_match_submap.get_frame_at_index(best_frame_index)
    
    # 确保两个张量尺寸一致
    if new_image_frame_tensor.shape[-2:] != best_match_frame.shape[-2:]:
        # 调整尺寸以匹配新图像
        best_match_frame = torch.nn.functional.interpolate(
            best_match_frame.unsqueeze(0),
            size=new_image_frame_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    best_match_frame_tensor = best_match_frame.to(device)
    
    # Concatenate the new image and the best match frame
    query_submap_images = torch.cat([
        new_image_frame_tensor, 
        best_match_frame_tensor.unsqueeze(0)
    ], dim=0)

    # Run VGGT model to get predictions
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(query_submap_images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], query_submap_images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    # Remove batch dimension and convert to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Step 3: Solve for the local-to-global transformation matrix
    # Get the point cloud of the best match frame in the query submap (from predictions)
    # Its coordinate system is relative to new_image (i.e. the local coordinate system of this prediction)
    if not solver.use_point_map:
        p_query_local = unproject_depth_map_to_point_map(predictions["depth"], predictions["extrinsic"], predictions["intrinsic"])[1].reshape(-1, 3)
        conf_query = predictions["depth_conf"][1].reshape(-1)
    else:
        p_query_local = predictions["world_points"][1].reshape(-1, 3)
        conf_query = predictions["world_points_conf"][1].reshape(-1)

    # Get the reference point cloud in the global coordinate system
    # a) get the point cloud of the best match frame in the reference submap (the local coordinate)
    p_ref_local = best_match_submap.get_frame_pointcloud(best_frame_index).reshape(-1, 3)
    conf_ref = best_match_submap.conf[best_frame_index].reshape(-1)
    # b) Get the global pose of the reference submap it belongs to
    H_world_ref = best_match_submap.get_reference_homography()
    # c) Transform the reference point cloud to the global coordinate system
    p_ref_world = apply_homography(H_world_ref, p_ref_local)
    
    # 确保两个点云长度一致
    min_length = min(len(p_query_local), len(p_ref_world))
    if min_length < 50:
        print(colored(f"Localization failed for {os.path.basename(new_image_path)}: Not enough point correspondences found ({min_length} points).", "red"))
        return None
    
    # 截取相同长度的点云
    p_query_local = p_query_local[:min_length]
    p_ref_world = p_ref_world[:min_length]
    conf_query = conf_query[:min_length]
    conf_ref = conf_ref[:min_length]
    
    # Create a mask for high-confidence points
    query_conf_threshold = np.percentile(conf_query, solver.init_conf_threshold)
    ref_conf_threshold = best_match_submap.get_conf_threshold()
    
    good_mask = (conf_query > query_conf_threshold) & (conf_ref > ref_conf_threshold)
    
    num_inliers = np.sum(good_mask)
    if num_inliers < 50: # set a minimum threshold for inliers
        print(colored(f"Localization failed for {os.path.basename(new_image_path)}: Not enough confident point correspondences found ({num_inliers} points).", "red"))
        return None

    # Step 4: Estimate the new global pose using RANSAC
    H_new = ransac_projective(p_query_local[good_mask], p_ref_world[good_mask])

    # Decompose the pose into rotation and translation
    vggt_intrinsic_new = predictions["intrinsic"][0]
    vggt_extrinsic_inv_new = np.linalg.inv(
        np.vstack([predictions["extrinsic"][0], [0, 0, 0, 1]])
    )

    R, t = decompose_pose_from_vggt(H_new, vggt_intrinsic_new, vggt_extrinsic_inv_new)
    return H_new, R, t


def get_image_files(directory):
    """获取目录下所有支持的图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def filter_outliers(method, points, **kwargs):
    """
    使用指定的方法过滤离群点
    
    Args:
        method (str): 离群点检测方法，可选值: 'zscore', 'mahalanobis', 'dbscan', 'isolation_forest', 'lof'
        points (np.ndarray): 点集，形状为 (N, 3)
        kwargs: 方法特定的参数
        
    Returns:
        np.ndarray: 布尔掩码，True 表示内点，False 表示离群点
    """
    if len(points) < 2:
        return np.ones(len(points), dtype=bool)
    
    if method == 'zscore':
        # Z-score 方法
        z_threshold = kwargs.get('z_threshold', 3.0)
        z_scores = scipy.stats.zscore(points, axis=0)
        return np.all(np.abs(z_scores) < z_threshold, axis=1)
    
    elif method == 'mahalanobis':
        # 马氏距离方法
        confidence_level = kwargs.get('confidence_level', 0.95)
        # 计算协方差矩阵的逆
        cov = np.cov(points.T)
        try:
            inv_cov = inv(cov)
        except:
            # 如果协方差矩阵不可逆，使用单位矩阵
            inv_cov = np.eye(3)
        
        # 计算均值
        mean = np.mean(points, axis=0)
        
        # 计算每个点到均值的马氏距离
        distances = np.array([mahalanobis(point, mean, inv_cov) for point in points])
        
        # 使用卡方分布确定阈值
        threshold = scipy.stats.chi2.ppf(confidence_level, df=3)  # 95%置信区间，3个自由度
        return distances < threshold
    
    elif method == 'dbscan':
        # DBSCAN 聚类方法
        eps = kwargs.get('eps', 0.1)
        min_samples = kwargs.get('min_samples', 3)
        
        # 使用 DBSCAN 聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # 找到最大的聚类
        cluster_counts = np.bincount(labels[labels >= 0])
        if len(cluster_counts) == 0:
            return np.ones(len(points), dtype=bool)
        
        largest_cluster = np.argmax(cluster_counts)
        return labels == largest_cluster
    
    elif method == 'isolation_forest':
        # 孤立森林方法
        contamination = kwargs.get('contamination', 0.1)  # 假设10%是离群点
        
        # 使用孤立森林检测离群点
        clf = IsolationForest(contamination=contamination, random_state=42)
        return clf.fit_predict(points) == 1
    
    elif method == 'lof':
        # 局部离群因子方法
        contamination = kwargs.get('contamination', 0.1)
        n_neighbors = kwargs.get('n_neighbors', 5)
        
        # 使用 LOF 检测离群点
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        return lof.fit_predict(points) == 1
    
    else:
        # 默认方法：保留所有点
        return np.ones(len(points), dtype=bool)

def main():
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Pre-loading dependent models (salad/dinov2) for deserialization...")
    try:
        # Pre-load salad model
        torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)
    except Exception as e:
        print(colored(f"Warning: Failed to pre-load salad model, deserialization might fail. Error: {e}", "yellow"))

    # Load the pre-built map state
    print(f"Loading pre-built map state from {args.map_path}...")
    if not os.path.exists(args.map_path):
        print(colored(f"Error: Map file not found at {args.map_path}", "red"))
        print("Please run main.py first to build and save a map.")
        return
        
    map_state = torch.load(args.map_path, map_location=device)
    print("Map state loaded successfully.")

    print("Reconstructing Solver for localization...")
    # Create a new Solver instance with the loaded map state
    solver = Solver(init_conf_threshold=map_state['init_conf_threshold'], use_point_map=map_state['use_point_map'], 
                    use_sim3=map_state['use_sim3'], gradio_mode=False)

    # Fill the new instance with the loaded core data
    solver.map = map_state['map']
    solver.image_retrieval = map_state['image_retrieval']
    solver.device = device

    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "/data/raid5/mayuying/models/VGGT-1B.pt"
    model.load_state_dict(torch.load(_URL))
    model.eval()
    model = model.to(device)
    
    # Start Viser server for visualization
    print("Updating visualization in existing Viser server...")
    solver.update_all_submap_vis()
    
    # Counter for localization sessions
    session_counter = 0
    
    # 用于存储可视化元素的句柄
    visualization_handles = {
        'cameras': [],
        'points': [],
        'avg_points': [],
        'labels': [],
        'lines': []
    }
    
    # Main loop for continuous localization
    while True:
        input("\nPress Enter to start a new localization session...")
        session_counter += 1
        print(f"\nStarting localization session #{session_counter}")
        
        # 清除之前的可视化结果
        for handle_list in visualization_handles.values():
            for handle in handle_list:
                try:
                    handle.remove()
                except:
                    pass
            handle_list.clear()
        
        # 从配置文件读取配置
        config_path = f"{args.localization_dir}/localization_config.txt"  # 配置文件路径
        outlier_method = "none"  # 默认方法
        method_params = {
            'z_threshold': 3.0,
            'confidence_level': 0.95,
            'eps': 0.1,
            'min_samples': 3,
            'contamination': 0.1,
            'n_neighbors': 5
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("outlier_method="):
                            outlier_method = line.split("=")[1].strip()
                            print(f"Using outlier_method={outlier_method} from config file")
                        elif line.startswith("z_threshold="):
                            method_params['z_threshold'] = float(line.split("=")[1].strip())
                        elif line.startswith("confidence_level="):
                            method_params['confidence_level'] = float(line.split("=")[1].strip())
                        elif line.startswith("eps="):
                            method_params['eps'] = float(line.split("=")[1].strip())
                        elif line.startswith("min_samples="):
                            method_params['min_samples'] = int(line.split("=")[1].strip())
                        elif line.startswith("contamination="):
                            method_params['contamination'] = float(line.split("=")[1].strip())
                        elif line.startswith("n_neighbors="):
                            method_params['n_neighbors'] = int(line.split("=")[1].strip())
            except Exception as e:
                print(colored(f"Error reading config file: {e}. Using default values", "yellow"))
        else:
            print("No config file found. Using default values")
        
        # Get all images in the directory
        image_paths = get_image_files(args.localization_dir+"/images")
        if not image_paths:
            print(colored(f"Error: No images found in {args.localization_dir}", "red"))
            continue
        
        print(f"Found {len(image_paths)} images for localization")
        
        # Run localization for all images
        translations = []
        valid_results = []
        
        for image_path in tqdm(image_paths, desc="Localizing images"):
            start_time = time.time()
            result = localize_image(solver, model, image_path)
            end_time = time.time()
            
            if result is not None:
                H_new, R, t = result
                translations.append(t)
                valid_results.append({
                    "image": os.path.basename(image_path),
                    "path": image_path,
                    "time": end_time - start_time,
                    "position": t,
                    "rotation": R
                })
        
        if not translations:
            print(colored("All localizations failed in this session.", "red"))
            continue
        
        # Filter outliers using the selected method
        translations_arr = np.array(translations)
        if len(translations_arr) > 1:
            good_mask = filter_outliers(outlier_method, translations_arr, **method_params)
            filtered_translations = translations_arr[good_mask]
            valid_results = [vr for i, vr in enumerate(valid_results) if good_mask[i]]
        else:
            filtered_translations = translations_arr
            good_mask = np.array([True])
        
        if len(filtered_translations) == 0:
            print(colored("All positions filtered as outliers. Using original average.", "yellow"))
            avg_translation = np.mean(translations_arr, axis=0)
        else:
            avg_translation = np.mean(filtered_translations, axis=0)
        
        print("\n" + "="*15 + " LOCALIZATION RESULT " + "="*15)
        print(f"Successfully localized {len(translations)}/{len(image_paths)} images")
        print(f"After outlier removal: {len(filtered_translations)} positions used for average")
        print(f"Used method: {outlier_method}")
        print(f"Average VGGT-SLAM position: x={avg_translation[0]:.4f}, y={avg_translation[1]:.4f}, z={avg_translation[2]:.4f}")
        print(f"Map to Odometry scale factor: {map_state['use_sim3']:.4f}")
        ros_position = map_state['use_sim3']*[avg_translation[2], -avg_translation[0], -avg_translation[1]]  # Convert to ROS coordinate system
        print(f"Average ROS position: x={ros_position[0]:.4f}, y={ros_position[1]:.4f}, z={ros_position[2]:.4f}")
        
        # Visualize the result in Viser
        print("Updating visualization in Viser...")
        
        # Display individual localized camera poses
        for i, res in enumerate(valid_results):
            se3_pose = np.eye(4)
            se3_pose[:3, :3] = res["rotation"]
            se3_pose[:3, 3] = res["position"]
            T_world_cam = viser_tf.SE3.from_matrix(se3_pose)

            cam_name = f"localized_cam/session_{session_counter}/{res['image']}"
            cam_handle = solver.viewer.server.scene.add_camera_frustum(
                name=cam_name,
                fov=np.pi / 3,
                aspect=1.0,
                scale=0.08,
                color=(220, 30, 30),  # Red
                line_width=1.5,
            )
            cam_handle.position = T_world_cam.translation()
            cam_handle.wxyz = T_world_cam.rotation().wxyz
            visualization_handles['cameras'].append(cam_handle)

            point_handle = solver.viewer.server.scene.add_icosphere(
                name=f"localized_point/session_{session_counter}/{res['image']}",
                radius=0.02,
                color=(0, 220, 0)    # Green
            )
            point_handle.position = T_world_cam.translation()
            visualization_handles['points'].append(point_handle)
            
            # Add text label for each camera
            label_handle = solver.viewer.server.scene.add_label(
                name=f"{cam_name}_label",
                text=f"{res['image']}",
                position=T_world_cam.translation() + np.array([0, 0, 0.05]),
            )
            visualization_handles['labels'].append(label_handle)
        
        # Display average position with a larger marker
        avg_point_name = f"avg_position/session_{session_counter}"
        avg_point_handle = solver.viewer.server.scene.add_icosphere(
            name=avg_point_name,
            radius=0.05,   # Larger size for visibility
            color=(0, 255, 0),  # Bright green
        )
        avg_point_handle.position = avg_translation
        visualization_handles['avg_points'].append(avg_point_handle)
        
        # Add text label for average position
        avg_label_handle = solver.viewer.server.scene.add_label(
            name=f"{avg_point_name}_label",
            text=f"Session #{session_counter} Avg",
            position=avg_translation + np.array([0, 0, 0.1]),
        )
        visualization_handles['labels'].append(avg_label_handle)
        
        # 添加连接所有点到平均位置的线
        for res in valid_results:
            line_name = f"line_to_avg/session_{session_counter}/{res['image']}"
            # 创建线段点数组：形状为(1, 2, 3)
            points = np.array([[res["position"], avg_translation]])
            # 创建颜色数组：形状为(1, 2, 3)
            colors = np.full((1, 2, 3), (255, 255, 0))  # Yellow for all points
            
            line_handle = solver.viewer.server.scene.add_line_segments(
                name=line_name,
                points=points,
                colors=colors,
                line_width=1.0,
            )
            visualization_handles['lines'].append(line_handle)
        
        print(colored(f"Average position visualized as large green sphere", "green"))
        print(colored(f"Session #{session_counter} completed. Press Enter to start a new session.", "blue"))

if __name__ == "__main__":
    main()
