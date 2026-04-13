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
        self.declare_parameter('min_frontier_size', 0.5)
        self.declare_parameter('planner_frequency', 0.2)
        self.declare_parameter('orientation_scale', 1.0)

        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.planner_freq = self.get_parameter('planner_frequency').value
        self.orientation_scale = self.get_parameter('orientation_scale').value

        self.get_logger().info(f'파라미터: potential_scale={self.get_parameter("potential_scale").value}')
        self.get_logger().info(f'파라미터: gain_scale={self.get_parameter("gain_scale").value}')
        self.get_logger().info(f'파라미터: min_frontier_size={self.get_parameter("min_frontier_size").value}')
        self.get_logger().info(f'파라미터: planner_frequency={self.get_parameter("planner_frequency").value}')
        self.get_logger().info(f'파라미터 로드 완료: topic={self.costmap_topic}, scale={self.potential_scale}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.explorer = FrontierSearch(
            potential_scale=self.potential_scale,
            gain_scale=self.gain_scale,
            min_frontier_size=self.min_frontier_size
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            self.costmap_topic, 
            self.map_callback, 
            10
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.last_map = None
        self.map_updated = False
        self.is_moving = False

        self.current_goal = None
        self.blocked_goals = []

        self.goal_handle = None
        self.nav_available = False


        self.last_fail_time = 0
        self.retry_interval = 5.0  # 5초 대기

        self.create_timer(0.5, self.check_nav2)  # Nav2 서버 상태 주기적으로 확인

        self.timer = self.create_timer(1.0 / self.planner_freq, self.plan_exploration)

        # self.hazard_sub = self.create_subscription(
        #     HazardDetectionVector,
        #     '/hazard_detection',
        #     self.bump_hit,
        #     qos
        # )

# =================================

    def get_robot_pose(self):
        """로봇의 위치를 가지고오는 함수"""
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 
                self.robot_base_frame, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            return (trans.transform.translation.x, trans.transform.translation.y)
        except TransformException as e:
            self.get_logger().error(f'로봇 위치를 가져오는 데 실패했습니다: {e}')
            return None

# =================================

    def map_callback(self, msg):
        self.last_map = msg
        self.map_updated = True

# =================================

    def plan_exploration(self):

        if not self.nav_available:
            self.get_logger().warn('Nav2가 없어 탐색 중지됨')
            return

        if time.time() - self.last_fail_time < self.retry_interval:
            return    

        if self.is_moving or self.last_map is None or not self.map_updated:
            self.get_logger().debug('탐색 계획 대기 중: 이동 중이거나 지도가 없음')
            return
        
        curr_pose = self.get_robot_pose()
        if curr_pose is None:
            self.get_logger().warn('로봇 현재 위치(tf)를 가져올 수 없어 탐색 계획을 건너뜁니다.')
            return
        
        self.map_updated = False

        frontiers = self.explorer.search_from(self.last_map, curr_pose)

        frontiers = [
            f for f in frontiers
            if self.is_safe_goal(f['world_point'])
        ]

        if not frontiers:
            self.get_logger().info('탐색 완료.')
            return
        
        target = frontiers[0]['world_point']
        
        if self.current_goal and self.is_same_goal(target, self.current_goal):
            self.get_logger().info('현재 목표와 동일한 지점이 탐색되었습니다. 새로운 목표로 이동하지 않습니다.')
            return
        
        if not frontiers or 'world_point' not in frontiers[0]:
            self.get_logger().warn('유효한 탐색 지점이 없습니다.')
            return

        self.get_logger().info(f'새로운 목표 지점 탐색: {target}, 비용: {frontiers[0]["cost"]:.3f}')

        self.send_goal(target, curr_pose)

# =================================

    def send_goal(self, target, curr_pose):
        
        self.is_moving = True
        self.current_goal = target

        if not self.nav_available:
            self.last_fail_time = time.time()
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Nav2 액션 서버에 연결할 수 없습니다. 목표를 보낼 수 없습니다.')
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

        goal_pose.pose.orientation.z = math.sin(yaw / 2)
        goal_pose.pose.orientation.w = math.cos(yaw / 2)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(f'목표 지점으로 이동 명령 준비: {target}')
        self.get_logger().info(f'Nav2로 이동 명령 전송: {target}')


        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

# =================================

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('목표가 거부되었습니다.')
            self.fail_current_goal()
            return
        
        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

# =================================

    def get_result_callback(self, future):
        self.is_moving = False

        result = future.result()
        status = result.status

        self.get_logger().info(f'future result: {status}')


        if status != 4:
            self.get_logger().warn(f'목표 달성 실패, 상태 코드: {status}')
            self.blocked_goals.append(self.current_goal)
            self.get_logger().info(f'목표 {self.current_goal} 차단 목록에 추가됨')
            self.last_fail_time = time.time()
            self.current_goal = None
            self.goal_handle = None
            return

        self.get_logger().info('목표 도착 완료')

# =================================

    def fail_current_goal(self):
        self.last_fail_time = time.time()
        self.current_goal = None
        self.is_moving = False

# =================================

    def is_same_goal(self, goal1, goal2, threshold=0.2):
        return math.hypot(goal1[0] - goal2[0], goal1[1] - goal2[1]) < threshold

# =================================

    def check_nav2(self):
        self.nav_available = self.nav_client.wait_for_server(timeout_sec=2.0)


# =================================

    def is_safe_goal(self, point):
        x, y = point

        info = self.last_map.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        index = my * info.width + mx
        cost = self.last_map.data[index]

        return cost == 0  # 0은 free space, -1은 unknown, 100 이상은 장애물
# =================================

    # def bump_hit(self, msg):
    #     self.get_logger().warn('충돌 감지! 현재 목표를 차단된 목록에 추가합니다.')
        
    #     for hazard in msg.detections:
    #         if hazard.type == hazard.BUMP:

    #             # self.get_logger().info(f'충돌 위치: {hazard.header.frame_id}, 거리: {hazard.range}')

    #             if self.current_goal and self.current_goal not in self.blocked_goals:
    #                 self.blocked_goals.append(self.current_goal)
    #                 self.get_logger().info(f'목표 {self.current_goal} 차단 목록에 추가됨')
    #                 self.current_goal = None  # 현재 목표 초기화
    #                 self.goal_handle = None
    #                 self.is_moving = True  # 이동 상태 초기화
    #                 self.bump_timer = self.create_timer(5.0, self.reset_bump_timer)  # 5초 후에 타이머 콜백 실행
    #             else:
    #                 self.get_logger().info('현재 목표가 없거나 이미 차단된 목록에 있습니다.')
    #                 self.bump_timer = self.create_timer(5.0, self.reset_bump_timer)  # 5초 후에 타이머 콜백 실행
    #             break
        

    # def reset_bump_timer(self):
    #     self.bump_timer.cancel()
    #     self.bump_timer.destroy()
    #     self.goal_handle.cancel_goal_async()
    #     self.bump_timer = None
    #     self.is_moving = True  # 타이머가 끝난 후 탐색 재시작 가능하도록 상태 초기화
    #     self.plan_exploration()  # 타이머가 끝나면 즉시 탐색 계획 실행


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
    
    