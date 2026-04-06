import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. YAML 파일 경로 설정
    # 패키지 안에 'config' 폴더를 만들고 그 안에 params_costmap.yaml을 넣었다고 가정합니다.
    config = os.path.join(
        get_package_share_directory('automatic_driving'),
        'config',
        'params_costmap.yaml'
    )

    return LaunchDescription([
        Node(
            package='automatic_driving',
            executable='automatic_driving_node', # setup.py에 정의된 이름
            name='automatic_driving',       # YAML의 최상단 이름과 반드시 일치!
            parameters=[config],       # 여기서 YAML 파일을 주입합니다.
            output='screen'
        )
    ])