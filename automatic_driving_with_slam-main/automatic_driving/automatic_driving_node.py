import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import TransformException, Buffer, TransformListener
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .frontier_search import FrontierSearch
from .costmap2dclient import Costmap2DClient

import math
import time


class Driving(Node):
    def __init__(self):
        super().__init__('automatic_driving')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('potential_scale', 1e-3)
        self.declare_parameter('gain_scale', 1.0)
        self.declare_parameter('min_frontier_size', 0.05)
        self.declare_parameter('planner_frequency', 0.5)
        self.declare_parameter('orientation_scale', 1.0)

        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.planner_freq = self.get_parameter('planner_frequency').value
        self.orientation_scale = self.get_parameter('orientation_scale').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.blocked_goals = []
        self.blocked_radius = 0.7
        self.blacklist_radius = 0.5

        self.branch_goal = None
        self.branch_radius = 1.5

        self.startpose = None
        self.current_goal = None

        self.exploration_finished = False
        self.returning_home = False

        self.is_moving = False

        self.timer = self.create_timer(1.0 / self.planner_freq, self.plan_exploration)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.costmap2dclient = Costmap2DClient(self, self.tf_buffer)
        self.frontier_search = FrontierSearch(
            self.potential_scale,
            self.gain_scale,
            self.min_frontier_size
        )

    # ===============================
    # 거리 계산
    # ===============================
    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ===============================
    # 블락된 목표 체크
    # ===============================
    def is_blocked_goal(self, target):
        for blocked in self.blocked_goals:
            if self.dist(target, blocked) < self.blocked_radius:
                return True
        return False

    # ===============================
    # frontier 선택 (벽까지 따라가기)
    # ===============================
    def select_frontier_until_wall(self, frontiers, curr_pose):
        valid = [f for f in frontiers if not self.is_blocked_goal(f["world_point"])]

        if not valid:
            return None

        # 현재 branch 유지
        if self.branch_goal is not None:
            same_branch = [
                f for f in valid
                if self.dist(f["world_point"], self.branch_goal) < self.branch_radius
            ]

            if same_branch:
                selected = min(
                    same_branch,
                    key=lambda f: self.dist(f["world_point"], curr_pose)
                )
                self.branch_goal = selected["world_point"]
                return selected

        # 새로운 branch 시작
        selected = min(
            valid,
            key=lambda f: self.dist(f["world_point"], curr_pose)
        )
        self.branch_goal = selected["world_point"]
        return selected

    # ===============================
    # 로봇 위치
    # ===============================
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1)
            )
            return (
                trans.transform.translation.x,
                trans.transform.translation.y
            )
        except TransformException as e:
            self.get_logger().error(f'TF 실패: {e}')
            return None

    # ===============================
    # 탐색 메인 루프
    # ===============================
    def plan_exploration(self):

        if self.exploration_finished:
            return

        if self.is_moving:
            return

        if not self.costmap2dclient.costmap_received:
            return

        curr_pose = self.get_robot_pose()
        if curr_pose is None:
            return

        if self.startpose is None:
            self.startpose = curr_pose
            self.get_logger().info(f"초기위치: {self.startpose}")

        frontiers = self.frontier_search.search_from(
            self.costmap2dclient,
            curr_pose
        )

        # frontier 없음 → 복귀
        if not frontiers:
            self.get_logger().info('탐색 종료 → 초기 위치로 복귀')

            self.returning_home = True
            self.send_goal(self.startpose, curr_pose)
            return

        selected = self.select_frontier_until_wall(frontiers, curr_pose)

        if selected is None:
            self.get_logger().info('유효 frontier 없음 → 복귀')

            self.returning_home = True
            self.send_goal(self.startpose, curr_pose)
            return

        target = selected["world_point"]

        self.get_logger().info(f'탐색 목표: {target}')

        self.send_goal(target, curr_pose)

    # ===============================
    # 목표 전송
    # ===============================
    def send_goal(self, target, curr_pose):

        self.is_moving = True
        self.current_goal = target

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.is_moving = False
            self.current_goal = None
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = target[0]
        goal_pose.pose.position.y = target[1]

        dx = target[0] - curr_pose[0]
        dy = target[1] - curr_pose[1]

        yaw = math.atan2(dy, dx)

        goal_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    # ===============================
    # goal 응답
    # ===============================
    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.fail_current_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    # ===============================
    # 결과 처리
    # ===============================
    def get_result_callback(self, future):

        self.is_moving = False
        result = future.result()
        status = result.status

        if status != 4:
            self.get_logger().warn(f'실패: {status}')

            if self.current_goal:
                self.blocked_goals.append(self.current_goal)

            self.branch_goal = None
            self.current_goal = None
            return

        self.get_logger().info('도착 성공')

        # 복귀 완료 → 종료
        if self.returning_home:
            self.get_logger().info('탐색 완전 종료')
            self.exploration_finished = True

            if self.timer:
                self.timer.cancel()

        self.current_goal = None

    # ===============================
    def fail_current_goal(self):
        self.is_moving = False
        self.current_goal = None


def main():
    rclpy.init()
    node = Driving()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
