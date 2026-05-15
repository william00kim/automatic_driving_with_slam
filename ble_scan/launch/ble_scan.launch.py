import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. YAML 파일 경로 설정
    # 패키지 안에 'config' 폴더를 만들고 그 안에 params_costmap.yaml을 넣었다고 가정합니다.
    config = os.path.join(
        get_package_share_directory('ble_scan'),
    )

    return LaunchDescription([
        Node(
            package='ble_scan',
            executable='ble_scan_node', # setup.py에 정의된 이름
            name='ble_scan',
            output='screen'
        )
    ])