import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import TransformException, Buffer, TransformListener

from .frontier_search import FrontierSearch
from .costmap2dclient import Costmap2DClient

import math
import time
import os
import subprocess


class Driving(Node):
    def __init__(self):
        super().__init__('automatic_driving')

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
        self.blacklist_radius = 0.7

        self.startpose = None
        self.current_goal = None
        self.branch_goal = None

        self.goal_handle = None
        self.result_future = None

        self.is_moving = False
        self.returning_home = False
        self.exploration_finished = False

        self.stuck_position = None
        self.stuck_check_time = None

        self.stuck_distance_threshold = 0.15
        self.stuck_timeout = 12.0

        self.save_dir = os.path.expanduser('~/maps/map_kr')
        os.makedirs(self.save_dir, exist_ok=True)

        self.traj_path = os.path.join(self.save_dir, 'trajectory.csv')
        self.traj_file = open(self.traj_path, 'w')
        self.traj_file.write('time,x,y\n')
        self.traj_file.flush()

        self.map_save_prefix = os.path.join(self.save_dir, 'map')

        self.timer = self.create_timer(
            1.0 / self.planner_freq,
            self.plan_exploration
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.costmap2dclient = Costmap2DClient(self, self.tf_buffer)

        self.frontier_search = FrontierSearch(
            self.potential_scale,
            self.gain_scale,
            self.min_frontier_size
        )

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
            self.get_logger().error(f'로봇 위치 가져오기 실패: {e}')
            return None

    def save_trajectory_point(self, curr_pose):
        now = self.get_clock().now().nanoseconds / 1e9
        self.traj_file.write(f'{now},{curr_pose[0]},{curr_pose[1]}\n')
        self.traj_file.flush()

    def plan_exploration(self):
        if self.exploration_finished:
            return

        if self.is_moving:
            return

        if not self.costmap2dclient.costmap_received:
            self.get_logger().warn('Costmap 아직 없음')
            return

        curr_pose = self.get_robot_pose()

        if self.is_moving:

            if self.stuck_position is None:
                self.stuck_position = curr_pose
                self.stuck_check_time = time.time()

            else:
                moved_dist = self.dist(curr_pose, self.stuck_position)

                # 거의 안 움직임
                if moved_dist < self.stuck_distance_threshold:

                    elapsed = time.time() - self.stuck_check_time

                    if elapsed > self.stuck_timeout:

                        self.get_logger().warn(
                            f'STUCK 감지 → 목표 포기: {self.current_goal}'
                        )

                        if self.current_goal is not None:
                            self.blocked_goals.append(self.current_goal)

                        self.branch_goal = None

                        # Nav2 goal cancel
                        if self.goal_handle is not None:
                            cancel_future = self.goal_handle.cancel_goal_async()

                        self.is_moving = False
                        self.current_goal = None

                        return

        else:
            # 움직였으면 갱신
            self.stuck_position = curr_pose
            self.stuck_check_time = time.time()

        if curr_pose is None:
            return

        if self.startpose is None:
            self.startpose = curr_pose
            self.get_logger().info(f'초기위치: {self.startpose}')

        self.save_trajectory_point(curr_pose)

        frontier = self.frontier_search.search_from(
            self.costmap2dclient,
            curr_pose
        )

        if not frontier:
            self.get_logger().info('탐색 종료 → 초기 위치로 복귀 시작')
            self.returning_home = True
            self.send_goal(self.startpose, curr_pose)
            return

        selected_frontier = self.select_frontier_until_wall(frontier, curr_pose)

        if selected_frontier is None:
            self.get_logger().info('이동 가능한 frontier 없음 → 초기 위치로 복귀 시작')
            self.returning_home = True
            self.send_goal(self.startpose, curr_pose)
            return

        target = selected_frontier['world_point']

        self.get_logger().info(
            f'현재 branch 계속 탐색 목표: {target}, 비용: {selected_frontier["cost"]:.3f}'
        )

        self.send_goal(target, curr_pose)

    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_blocked_goal(self, target):
        for blocked in self.blocked_goals:
            if self.dist(target, blocked) < self.blacklist_radius:
                return True
        return False

    def select_frontier_until_wall(self, frontiers, curr_pose):
        valid_frontiers = []

        for f in frontiers:
            target = f['world_point']

            if not self.is_blocked_goal(target):
                valid_frontiers.append(f)

        if not valid_frontiers:
            return None

        if self.branch_goal is not None:
            same_branch = []

            for f in valid_frontiers:
                target = f['world_point']

                if self.dist(target, self.branch_goal) < 1.5:
                    same_branch.append(f)

            if same_branch:
                selected = min(
                    same_branch,
                    key=lambda f: self.dist(f['world_point'], curr_pose)
                )
                self.branch_goal = selected['world_point']
                return selected

        selected = min(
            valid_frontiers,
            key=lambda f: self.dist(f['world_point'], curr_pose)
        )

        self.branch_goal = selected['world_point']
        return selected

    def send_goal(self, target, curr_pose):
        self.is_moving = True
        self.current_goal = target

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server에 연결할 수 없습니다.')
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

        goal_pose.pose.orientation.z = math.sin(yaw / 2.0) * self.orientation_scale
        goal_pose.pose.orientation.w = math.cos(yaw / 2.0) * self.orientation_scale

        self.get_logger().info(f'Nav2로 이동 명령 전송: {target}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Nav2가 목표를 거부했습니다.')
            self.fail_current_goal()
            return

        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.is_moving = False

        result = future.result()
        status = result.status

        self.get_logger().info(f'future result: {status}')

        if status != 4:
            self.get_logger().warn(f'목표 달성 실패, 상태 코드: {status}')

            if self.current_goal is not None:
                self.blocked_goals.append(self.current_goal)
                self.get_logger().info(f'목표 {self.current_goal} 차단 목록에 추가됨')

            self.branch_goal = None
            self.current_goal = None
            self.goal_handle = None
            return

        self.get_logger().info('목표 도착 완료')

        curr_pose = self.get_robot_pose()
        if curr_pose is not None:
            self.save_trajectory_point(curr_pose)

        if self.returning_home:
            self.get_logger().info('초기 위치 복귀 완료 → 지도와 경로 저장 시작')

            self.exploration_finished = True
            self.returning_home = False

            if self.timer is not None:
                self.timer.cancel()

            self.finish_and_save()

        self.current_goal = None
        self.goal_handle = None

    def finish_and_save(self):
        if self.traj_file is not None:
            self.traj_file.flush()
            self.traj_file.close()
            self.traj_file = None

        self.save_map()
        self.make_map_with_trajectory()

        self.get_logger().info(f'저장 완료: {self.save_dir}')

    def save_map(self):
        try:
            cmd = [
                'ros2',
                'run',
                'nav2_map_server',
                'map_saver_cli',
                '-f',
                self.map_save_prefix
            ]

            self.get_logger().info('map_saver_cli 실행 중...')
            subprocess.run(cmd, check=True)
            self.get_logger().info('지도 저장 완료')

        except Exception as e:
            self.get_logger().error(f'지도 저장 실패: {e}')

    def make_map_with_trajectory(self):
        try:
            import yaml
            import pandas as pd
            import matplotlib.pyplot as plt
            from PIL import Image

            yaml_path = self.map_save_prefix + '.yaml'
            pgm_path = self.map_save_prefix + '.pgm'
            output_path = os.path.join(self.save_dir, 'map_with_trajectory.png')

            with open(yaml_path, 'r') as f:
                map_info = yaml.safe_load(f)

            resolution = map_info['resolution']
            origin_x, origin_y, _ = map_info['origin']

            img = Image.open(pgm_path)
            width, height = img.size

            traj = pd.read_csv(self.traj_path)

            if len(traj) < 2:
                self.get_logger().warn('trajectory 점이 부족해서 경로 이미지를 만들 수 없습니다.')
                return

            traj['px'] = (traj['x'] - origin_x) / resolution
            traj['py'] = height - ((traj['y'] - origin_y) / resolution)

            plt.figure(figsize=(10, 10))
            plt.imshow(img, cmap='gray')
            plt.plot(traj['px'], traj['py'], linewidth=2)
            plt.scatter(traj['px'].iloc[0], traj['py'].iloc[0], s=80, label='Start')
            plt.scatter(traj['px'].iloc[-1], traj['py'].iloc[-1], s=80, label='End')
            plt.legend()
            plt.axis('equal')
            plt.savefig(output_path, dpi=300)
            plt.close()

            self.get_logger().info(f'경로 포함 지도 저장 완료: {output_path}')

        except Exception as e:
            self.get_logger().error(f'경로 지도 생성 실패: {e}')

    def fail_current_goal(self):
        self.current_goal = None
        self.is_moving = False

    def destroy_node(self):
        if hasattr(self, 'traj_file') and self.traj_file is not None:
            self.traj_file.close()
            self.traj_file = None

        super().destroy_node()


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