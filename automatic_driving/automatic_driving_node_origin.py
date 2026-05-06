import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import TransformException, Buffer, TransformListener
# from irobot_create_msgs.msg import HazardDetectionVector
from rclpy.qos import QoSProfile, ReliabilityPolicy

# 같은 패키지 내의 frontier_search.py에서 클래스 임포트
from .frontier_search import FrontierSearch 
from .costmap2dclient import Costmap2DClient 

import math
import time

class Driving(Node):
    def __init__(self):
        # YAML의 'explore_node' 또는 'automatic_driving' 세션 이름과 일치해야 함
        super().__init__('automatic_driving') 

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('potential_scale', 1e-3)
        self.declare_parameter('gain_scale', 1.0)
        self.declare_parameter('min_frontier_size', 0.05)
        self.declare_parameter('planner_frequency', 1.0)
        self.declare_parameter('orientation_scale', 1.0)

        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.planner_freq = self.get_parameter('planner_frequency').value
        self.orientation_scale = self.get_parameter('orientation_scale').value

        self.get_logger().info(f'파라미터: robot_base_frame={self.get_parameter("robot_base_frame").value}')
        self.get_logger().info(f'파라미터: potential_scale={self.get_parameter("potential_scale").value}')
        self.get_logger().info(f'파라미터: gain_scale={self.get_parameter("gain_scale").value}')
        self.get_logger().info(f'파라미터: min_frontier_size={self.get_parameter("min_frontier_size").value}')
        self.get_logger().info(f'파라미터: planner_frequency={self.get_parameter("planner_frequency").value}')
        self.get_logger().info(f'파라미터: orientation_scale={self.get_parameter("orientation_scale").value}')
        self.get_logger().info(f'파라미터 로드 완료: topic={self.costmap_topic}, scale={self.potential_scale}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.blocked_goals = []

        self.startpose = None
        self.current_goal = None

        self.startpose = None
        self.current_goal = None
        self.branch_goal = None

        self.goal_handle = None
        self.result_future = None

        self.blacklist_radius = 0.7

        self.is_moving = False

        self.timer = self.create_timer(1.0 / self.planner_freq, self.plan_exploration)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.costmap2dclient = Costmap2DClient(self, self.tf_buffer)

        self.frontier_search = FrontierSearch(self.potential_scale, self.gain_scale,self.min_frontier_size)

        self.traj_file = open('~/maps/map_kr/trajectory.csv', 'w')
        self.traj_file.write('time,x,y\n')

    # =================================
    # 로봇 위치 가져오기 함수
    # =================================

    def get_robot_pose(self):
        self.get_logger().debug('로봇 위치 가져오기 시도')
        try:
            trans = self.tf_buffer.lookup_transform('map', self.robot_base_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1))
            return (trans.transform.translation.x, trans.transform.translation.y)
        except TransformException as e:
            self.get_logger().error(f'로봇 위치 가져오기 실패: {e}')
            return None
        
    # =================================
    # 탐색 계획 함수
    # =================================

    def plan_exploration(self):
        
        if self.is_moving:
            self.get_logger().debug('현재 이동 중이므로 탐색 계획 생략')
            return
        
        if not self.costmap2dclient.costmap_received:
            self.get_logger().warn("Costmap 아직 없음")
            return

        if not self.nav_client:
            self.get_logger().error('Nav2 action client is not available.')
            return

        curr_pose = self.get_robot_pose()

        if curr_pose is None:
            self.get_logger().error('현재 로봇 위치(tf)를 가져올 수 없습니다.')
            return
        
        if self.startpose is None:
            self.startpose = curr_pose
            self.get_logger().info(f"초기위치: {self.startpose}")

        now = self.get_clock().now().nanoseconds / 1e9
        self.traj_file.write(f'{now},{curr_pose[0]},{curr_pose[1]}\n')
        self.traj_file.flush()

        self.map_updated = False

        frontier = self.frontier_search.search_from(self.costmap2dclient, curr_pose)

        # self.get_logger().info(f'frontier: {frontier}')

        if not frontier:
            self.get_logger().info('탐색 종료 → 초기 위치로 복귀')
            self.send_goal(self.startpose, curr_pose)
            return

        selected_frontier = self.select_depth_first_frontier(frontier, curr_pose)

        if selected_frontier is None:
            self.get_logger().info('이동 가능한 frontier 없음 → 탐색 종료 또는 초기 위치 복귀')
            self.send_goal(self.startpose, curr_pose)
            return

        target = selected_frontier["world_point"]

        self.get_logger().info(
            f'DFS 방식 목표 선택: {target}, 비용: {selected_frontier["cost"]:.3f}'
        )

        self.send_goal(target, curr_pose)


    def is_blocked_goal(self, target):
        tx, ty = target

        for bx, by in self.blocked_goals:
            dist = math.sqrt((tx - bx) ** 2 + (ty - by) ** 2)

            if dist < self.blacklist_radius:
                return True

        return False


    def select_depth_first_frontier(self, frontiers, curr_pose):
        valid_frontiers = []

        for f in frontiers:
            target = f["world_point"]

            if not self.is_blocked_goal(target):
                valid_frontiers.append(f)

        if not valid_frontiers:
            return None

        # 현재 branch_goal 주변 frontier를 우선 선택
        if self.branch_goal is not None:
            bx, by = self.branch_goal

            same_branch = []

            for f in valid_frontiers:
                x, y = f["world_point"]
                dist_from_branch = math.sqrt((x - bx) ** 2 + (y - by) ** 2)

                if dist_from_branch < 2.0:
                    same_branch.append(f)

            if same_branch:
                # 같은 branch 안에서는 현재 위치에서 더 먼 frontier 선택
                return max(
                    same_branch,
                    key=lambda f: math.sqrt(
                        (f["world_point"][0] - curr_pose[0]) ** 2 +
                        (f["world_point"][1] - curr_pose[1]) ** 2
                    )
                )

        # 처음이거나 기존 branch가 막혔으면 가장 먼 frontier 선택
        selected = max(
            valid_frontiers,
            key=lambda f: math.sqrt(
                (f["world_point"][0] - curr_pose[0]) ** 2 +
                (f["world_point"][1] - curr_pose[1]) ** 2
            )
        )

        self.branch_goal = selected["world_point"]
        return selected

    # ========================================
    # 목표 전송 및 Nav2 액션 클라이언트 콜백
    # ========================================

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

        # z축 기준으로 회전하기 때문에 x, y가 변화
        yaw = math.atan2(dy, dx)

        goal_pose.pose.orientation.z = math.sin(yaw / 2.0) * self.orientation_scale
        goal_pose.pose.orientation.w = math.cos(yaw / 2.0) * self.orientation_scale
        
        self.get_logger().info(f'목표 지점으로 이동 명령 준비: {target}')
        self.get_logger().info(f'Nav2로 이동 명령 전송: {target}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose


        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)


    # ========================================
    # Nav2 액션 클라이언트 콜백: 목표 응답 처리
    # ========================================

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Nav2가 목표를 거부했습니다.')
            self.fail_current_goal()
            self.is_moving = False
            return
        
        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)


    # ========================================
    # Nav2 액션 클라이언트 콜백: 목표 결과 처리
    # ========================================

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
            self.last_fail_time = time.time()
            self.current_goal = None
            self.goal_handle = None

            return

        self.get_logger().info('목표 도착 완료')
        self.current_goal = None
        self.goal_handle = None
    # =================================
    # 목표 실패 처리 함수
    # =================================

    def fail_current_goal(self):
        self.last_fail_time = time.time()
        self.current_goal = None
        self.is_moving = False


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
    
    