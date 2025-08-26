#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Path Planner with Periodic Path Publishing
-------------------------------------------------
1) Listens to /navigation_control for target points
2) Projects start and end points to simplified map
3) Finds shortest path between projected points
4) Publishes path to /planner/target_path immediately and every 5 seconds
5) Visualizes simplified map and planned path with timestamp
6) Automatically creates output directory if needed
"""
import argparse, json, os, math, threading, traceback, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from typing import List, Tuple
from datetime import datetime

# ---------- ROS2 ----------
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String as StringMsg
except ImportError:
    rclpy = None

# ---------- WGS-84 to ENU ----------
_A = 6378137.0; _F = 1/298.257223563; _E2 = _F*(2-_F)

def lla_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Convert GPS to ECEF"""
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    s, c = math.sin(lat_r), math.cos(lat_r)
    N = _A / math.sqrt(1 - _E2 * s * s)
    x = (N + h) * c * math.cos(lon_r)
    y = (N + h) * c * math.sin(lon_r)
    z = (N * (1 - _E2) + h) * s
    return np.array([x, y, z])

def ecef_to_enu(xyz: np.ndarray, lat0: float, lon0: float, h0: float) -> np.ndarray:
    """Convert ECEF to ENU"""
    ref = lla_to_ecef(lat0, lon0, h0)
    lat0_r, lon0_r = math.radians(lat0), math.radians(lon0)
    sL, cL, sλ, cλ = math.sin(lat0_r), math.cos(lat0_r), math.sin(lon0_r), math.cos(lon0_r)
    R = np.array([[-sλ, cλ, 0],
                  [-sL*cλ, -sL*sλ, cL],
                  [cL*cλ, cL*sλ, sL]])
    return R.dot(xyz - ref)

def lla_to_enu(lat: float, lon: float, h: float,
               lat0: float, lon0: float, h0: float) -> np.ndarray:
    """Convert GPS to ENU"""
    return ecef_to_enu(lla_to_ecef(lat, lon, h), lat0, lon0, h0)

# ---------- Read Simplified Map ----------
def read_simplified_map(map_path: str) -> pd.DataFrame:
    """Load simplified map from CSV file"""
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")
    
    if map_path.endswith(".csv"):
        df = pd.read_csv(map_path)
        # Ensure column names
        if "latitude" not in df.columns or "longitude" not in df.columns:
            if "lat" in df.columns and "lon" in df.columns:
                df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
            else:
                raise ValueError("Map file must contain 'latitude' and 'longitude' columns")
        return df
    else:
        raise ValueError("Unsupported map format. Only CSV files are supported")

# ---------- Project Points to Map ----------
def project_to_map(
    points: List[Tuple[float, float]],
    map_points: pd.DataFrame,
    max_distance: float = 50.0
) -> List[Tuple[float, float]]:
    """
    Project points to the closest points on the map
    :param points: List of (lat, lon) tuples to project
    :param map_points: DataFrame with 'latitude' and 'longitude' columns
    :param max_distance: Maximum projection distance (m)
    :return: List of projected (lat, lon) tuples
    """
    if len(map_points) == 0:
        return points
    
    # Calculate reference point
    lat0 = map_points["latitude"].iloc[0]
    lon0 = map_points["longitude"].iloc[0]
    
    # Convert map points to ENU
    map_enu = []
    for _, row in map_points.iterrows():
        enu = lla_to_enu(row["latitude"], row["longitude"], 0, lat0, lon0, 0)
        map_enu.append(enu[:2])
    
    # Build KDTree for efficient search
    tree = KDTree(map_enu)
    
    projected_points = []
    for lat, lon in points:
        # Convert point to ENU
        point_enu = lla_to_enu(lat, lon, 0, lat0, lon0, 0)[:2]
        
        # Find closest map point
        dist, idx = tree.query(point_enu)
        
        # Check if within max distance
        if dist <= max_distance:
            row = map_points.iloc[idx]
            projected_points.append((row["latitude"], row["longitude"]))
        else:
            # Use original point if too far
            projected_points.append((lat, lon))
    
    return projected_points

# ---------- Shortest Path Algorithm ----------
def dijkstra_shortest_path(
    map_points: pd.DataFrame,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    connection_radius: float = 20.0
) -> List[Tuple[float, float]]:
    """
    Find shortest path between start and end points on the map
    :param map_points: DataFrame with 'latitude' and 'longitude' columns
    :param start_point: (lat, lon) tuple of start point
    :param end_point: (lat, lon) tuple of end point
    :param connection_radius: Maximum connection distance between points (m)
    :return: List of (lat, lon) tuples representing the shortest path
    """
    if len(map_points) < 2:
        return []
    
    # Calculate reference point
    lat0 = map_points["latitude"].iloc[0]
    lon0 = map_points["longitude"].iloc[0]
    
    # Convert map points to ENU
    map_enu = []
    for _, row in map_points.iterrows():
        enu = lla_to_enu(row["latitude"], row["longitude"], 0, lat0, lon0, 0)
        map_enu.append(enu[:2])
    
    # Find indices of start and end points
    tree = KDTree(map_enu)
    _, start_idx = tree.query(lla_to_enu(start_point[0], start_point[1], 0, lat0, lon0, 0)[:2])
    _, end_idx = tree.query(lla_to_enu(end_point[0], end_point[1], 0, lat0, lon0, 0)[:2])
    
    # Build graph
    n = len(map_enu)
    row_ind = []
    col_ind = []
    data = []
    
    for i in range(n):
        # Find neighbors within connection radius
        neighbors = tree.query_ball_point(map_enu[i], connection_radius)
        
        for j in neighbors:
            if i == j:
                continue
            dist = math.sqrt((map_enu[i][0]-map_enu[j][0])**2 + (map_enu[i][1]-map_enu[j][1])**2)
            row_ind.append(i)
            col_ind.append(j)
            data.append(dist)
    
    # Create CSR matrix
    graph_csr = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    # Run Dijkstra's algorithm
    dist_matrix, predecessors = dijkstra(
        graph_csr, 
        directed=False, 
        indices=start_idx, 
        return_predecessors=True
    )
    
    # Reconstruct path
    path = []
    current = end_idx
    while current != start_idx:
        row = map_points.iloc[current]
        path.append((row["latitude"], row["longitude"]))
        current = predecessors[current]
        if current == -9999:  # No path found
            return []
    
    # Add start point
    row = map_points.iloc[start_idx]
    path.append((row["latitude"], row["longitude"]))
    
    return list(reversed(path))

# ---------- Publish Path to Topic (Enhanced Format) ----------
def publish_path_to_topic(
    path_points: List[Tuple[float, float]],
    topic: str = "/planner/target_path",
    action: int = 1,
    mode:int = 1,
    batch_id: str = "batch_auto"
):
    """Publish planned path to ROS topic in specified format"""
    if rclpy is None:
        raise ImportError("ROS2 Python packages not available")
    
    # Create JSON payload with specified format
    payload = {
        "action": action,
        "mode": mode,
        "batchId": batch_id,
        "points": [{"latitude": lat, "longitude": lon} for lat, lon in path_points]
    }
    json_str = json.dumps(payload)
    
    # Publish to topic
    # 检查是否已经初始化
    if not rclpy.ok():
        rclpy.init()
    
    node = Node("path_publisher")
    publisher = node.create_publisher(StringMsg, topic, 10)
    msg = StringMsg()
    msg.data = json_str
    
    # 发布消息
    publisher.publish(msg)
    
    # 确保消息发送
    rclpy.spin_once(node, timeout_sec=0.1)
    
    # 清理节点
    node.destroy_node()

# ---------- Generate Timestamped Filename ----------
def generate_timestamped_filename(base_path: str, extension: str = "") -> str:
    """
    Generate filename with timestamp
    :param base_path: Base file path (without extension)
    :param extension: File extension (e.g., ".png")
    :return: Timestamped filename
    """
    # Generate timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Split filename and extension
    base, ext = os.path.splitext(base_path)
    if not ext and extension:
        ext = extension
    
    # Create timestamped filename
    timestamped_path = f"{base}_{timestamp}{ext}"
    
    return timestamped_path

# ---------- Visualization (Enhanced) ----------
def visualize_path(
    map_points: pd.DataFrame,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    path_points: List[Tuple[float, float]],
    output_png: str = "path_visualization.png"
):
    """Visualize simplified map and planned path with timestamp in filename"""
    # Create directory if needed
    output_dir = os.path.dirname(output_png) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    if len(map_points) == 0:
        print("No map points to visualize")
        return
    
    # Calculate reference point
    lat0 = map_points["latitude"].iloc[0]
    lon0 = map_points["longitude"].iloc[0]
    
    # Convert all points to ENU
    map_enu, start_enu, end_enu, path_enu = [], [], [], []
    
    # Convert map points
    for _, row in map_points.iterrows():
        enu = lla_to_enu(row["latitude"], row["longitude"], 0, lat0, lon0, 0)
        map_enu.append(enu[:2])
    
    # Convert start and end points
    start_enu = lla_to_enu(start_point[0], start_point[1], 0, lat0, lon0, 0)[:2]
    end_enu = lla_to_enu(end_point[0], end_point[1], 0, lat0, lon0, 0)[:2]
    
    # Convert path points
    for lat, lon in path_points:
        enu = lla_to_enu(lat, lon, 0, lat0, lon0, 0)
        path_enu.append(enu[:2])
    
    # Plot map points (light gray)
    plt.scatter(
        [e[0] for e in map_enu], 
        [e[1] for e in map_enu],
        c="lightgray", marker=".", s=10, alpha=0.5, label="Simplified Map"
    )
    
    # Plot start and end points
    plt.scatter(
        start_enu[0], start_enu[1],
        c="red", marker="o", s=100, zorder=10, label="Start Point"
    )
    plt.scatter(
        end_enu[0], end_enu[1],
        c="blue", marker="o", s=100, zorder=10, label="End Point"
    )
    
    # Plot planned path
    if path_enu:
        plt.plot(
            [e[0] for e in path_enu], 
            [e[1] for e in path_enu],
            "g-", linewidth=3, alpha=0.8, zorder=5, label="Planned Path"
        )
        plt.scatter(
            [e[0] for e in path_enu], 
            [e[1] for e in path_enu],
            c="green", marker="o", s=40, zorder=6, alpha=0.7
        )
    
    # Add labels
    plt.text(start_enu[0], start_enu[1], "Start", fontsize=10, ha='right')
    plt.text(end_enu[0], end_enu[1], "End", fontsize=10, ha='left')
    
    # Configure plot
    plt.title("Path Planning Visualization", fontsize=16)
    plt.xlabel("East (m)", fontsize=14)
    plt.ylabel("North (m)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.axis("equal")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Visualization saved: {output_png}")

# ---------- Path Planner Node (Enhanced) ----------
class PathPlannerNode(Node):
    def __init__(self, map_path: str, max_projection: float = 50.0, 
                 connection_radius: float = 20.0, output_plot: str = "path_visualization.png"):
        super().__init__("path_planner_node")
        self.map_path = map_path
        self.max_projection = max_projection
        self.connection_radius = connection_radius
        self.output_plot = output_plot
        self.map_points = None
        self.path_points = []  # 存储当前路径点
        self.publish_timer = None  # 定时器
        self.current_action = 1
        self.current_mode = 1
        self.current_batch_id = "batch_auto"
        
        # Load simplified map
        self.load_map()
        
        # Subscribe to navigation control
        self.subscription = self.create_subscription(
            StringMsg,
            '/navigation_control',
            self.callback,
            10
        )
        self.get_logger().info("Listening to /navigation_control...")
    
    def load_map(self):
        """Load simplified map from file"""
        try:
            self.map_points = read_simplified_map(self.map_path)
            self.get_logger().info(f"Loaded {len(self.map_points)} map points")
        except Exception as e:
            self.get_logger().error(f"Error loading map: {str(e)}")
            self.map_points = pd.DataFrame()
    
    def start_periodic_publishing(self):
        """Start periodic publishing of path"""
        # 如果已经有定时器，先取消
        if self.publish_timer is not None:
            self.publish_timer.cancel()
        
        # 创建新定时器，每5秒发布一次
        self.publish_timer = self.create_timer(
            5.0,  # 5秒间隔
            self.publish_current_path
        )
        self.get_logger().info("Started periodic path publishing (every 5 seconds)")
    
    def publish_current_path(self):
        """Publish current path to topic"""
        if not self.path_points:
            self.get_logger().warn("No path to publish")
            return
        
        try:
            publish_path_to_topic(
                self.path_points,
                "/planner/target_path",
                self.current_action,
                self.current_mode,
                self.current_batch_id
            )
            self.get_logger().info("Periodic path published to /planner/target_path")
        except Exception as e:
            self.get_logger().error(f"Error publishing path: {str(e)}")
            self.get_logger().error(traceback.format_exc())
    
    def callback(self, msg: StringMsg):
        """Handle incoming navigation command"""
        try:
            self.get_logger().info(f"Received message: {msg.data[:100]}...")
            
            # Parse JSON data
            data = json.loads(msg.data)
            points = data.get("points", [])
            self.current_action = data.get("action", 1)
            self.current_mode = data.get("mode", 1)
            self.current_batch_id = data.get("batchId", "batch_auto")
            
            if len(points) < 2:
                self.get_logger().warn("Received less than 2 points")
                return
            
            # Extract start and end points
            start_point = (points[0].get("latitude", 0.0), points[0].get("longitude", 0.0))
            end_point = (points[-1].get("latitude", 0.0), points[-1].get("longitude", 0.0))
            
            self.get_logger().info(f"Start point: {start_point}")
            self.get_logger().info(f"End point: {end_point}")
            
            # Project points to map
            projected_points = project_to_map(
                [start_point, end_point], 
                self.map_points, 
                self.max_projection
            )
            projected_start = projected_points[0]
            projected_end = projected_points[1]
            
            self.get_logger().info(f"Projected start: {projected_start}")
            self.get_logger().info(f"Projected end: {projected_end}")
            
            # Find shortest path
            path_points = dijkstra_shortest_path(
                self.map_points,
                projected_start,
                projected_end,
                self.connection_radius
            )
            
            if not path_points:
                self.get_logger().error("No path found")
                return
            
            self.path_points = path_points  # 存储当前路径
            self.get_logger().info(f"Found path with {len(path_points)} points")
            
            # Publish to topic immediately
            try:
                publish_path_to_topic(
                    path_points,
                    "/planner/target_path",
                    self.current_action,
                    self.current_mode,
                    self.current_batch_id
                )
                self.get_logger().info("Path published to /planner/target_path")
            except Exception as e:
                self.get_logger().error(f"Error publishing path: {str(e)}")
                self.get_logger().error(traceback.format_exc())
            
            # Start periodic publishing
            self.start_periodic_publishing()
            
            # Create visualization (without projected points)
            try:
                # Generate timestamped filename
                timestamped_plot = generate_timestamped_filename(self.output_plot, ".png")
                self.get_logger().info(f"Creating visualization: {timestamped_plot}")
                
                visualize_path(
                    self.map_points,
                    start_point,
                    end_point,
                    path_points,
                    timestamped_plot
                )
            except Exception as e:
                self.get_logger().error(f"Error creating visualization: {str(e)}")
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON decode error: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error processing message: {str(e)}")
            self.get_logger().error(traceback.format_exc())

# ---------- Main Function (Continuous Operation) ----------
def main():
    parser = argparse.ArgumentParser(description="ROS2 Path Planner with Continuous Operation")
    parser.add_argument("--map", required=True, help="Path to simplified map CSV file")
    parser.add_argument("--max_projection", type=float, default=50.0,
                        help="Maximum projection distance (m)")
    parser.add_argument("--connection_radius", type=float, default=20.0,
                        help="Maximum connection radius between points (m)")
    parser.add_argument("--output_plot", default="path_visualization.png",
                        help="Output visualization file (will have timestamp added)")
    args = parser.parse_args()

    if rclpy is None:
        print("Error: ROS2 Python packages not available")
        return
    
    # Initialize ROS
    rclpy.init()
    
    # Create and run node
    node = PathPlannerNode(
        args.map,
        args.max_projection,
        args.connection_radius,
        args.output_plot
    )
    
    # Run in separate thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    # Run continuously until interrupted
    print("Path planner running. Press Ctrl+C to exit.")
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        print("Node shutdown")

if __name__ == "__main__":
    main()
