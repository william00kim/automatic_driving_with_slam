import numpy as np
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped
import rclpy
from tf2_ros import TransformException

class Costmap2DClient:
    def __init__(self, node, tf_buffer):
        self.node = node
        self.tf_buffer = tf_buffer

        self.global_frame = None
        self.robot_base_frame = 'base_link'

        self.costmap = None
        self.resolution = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0

        self.info = None
        self.data = None
        self.costmap_received = False

        # costmap 구독
        self.node.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.full_map_callback,
            10
        )

        # partial update 구독
        self.node.create_subscription(
            OccupancyGridUpdate,
            '/global_costmap/costmap_updates',
            self.partial_map_callback,
            10
        )

    # =====================================
    # cost 변환 (m-explore-lite 핵심)
    # =====================================
    def translate_cost(self, value):
        if value == 0:
            return 0  # free
        elif value == 100:
            return 254  # obstacle
        elif value == -1:
            return 255  # unknown
        else:
            return int(1 + (251 * (value - 1)) / 97)

    # =====================================
    # full map update
    # =====================================
    def full_map_callback(self, msg):
        #self.node.get_logger().info("Full costmap received")

        self.global_frame = msg.header.frame_id
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.info = msg.info

        self.data = np.array(msg.data, dtype=np.int16).reshape((self.height, self.width))
        # self.node.get_logger().info(f"{self.data}")

        # cost 변환 적용
        vectorized = np.vectorize(self.translate_cost)
        self.costmap = vectorized(self.data)

        self.costmap_received = True

    # =====================================
    # partial map update
    # =====================================
    def partial_map_callback(self, msg):
        if self.costmap is None:
            return

        x0 = msg.x
        y0 = msg.y
        w = msg.width
        h = msg.height

        data = np.array(msg.data, dtype=np.int16).reshape((h, w))
        vectorized = np.vectorize(self.translate_cost)
        updated = vectorized(data)

        self.costmap[y0:y0+h, x0:x0+w] = updated

    # =====================================
    # 로봇 위치 가져오기 (안정 버전)
    # =====================================
    def get_robot_pose(self):
        if self.global_frame is None:
            return None

        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )

            return (
                trans.transform.translation.x,
                trans.transform.translation.y
            )

        except TransformException as e:
            self.node.get_logger().warn(f'TF 실패: {e}')
            return None