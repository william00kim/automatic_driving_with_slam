import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import tf2_ros
import csv
import os
from datetime import datetime


class PosePathLogger(Node):
    def __init__(self):
        super().__init__('pose_path_logger')

        self.global_frame = 'map'
        self.robot_frame = 'base_link'

        self.save_dir = os.path.expanduser('~/turtlebot4_path_logs')
        os.makedirs(self.save_dir, exist_ok=True)

        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.save_dir, f'path_{now}.csv')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time_sec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path_pub = self.create_publisher(Path, '/saved_path', 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.global_frame

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info(f'Path logger started')
        self.get_logger().info(f'Saving CSV to: {self.csv_path}')

    def timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            t = trans.header.stamp.sec + trans.header.stamp.nanosec * 1e-9

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            qx = trans.transform.rotation.x
            qy = trans.transform.rotation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w

            self.writer.writerow([t, x, y, z, qx, qy, qz, qw])
            self.csv_file.flush()

            pose = PoseStamped()
            pose.header = trans.header
            pose.header.frame_id = self.global_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_msg.poses.append(pose)

            self.path_pub.publish(self.path_msg)

            self.get_logger().info(f'Current pose: x={x:.2f}, y={y:.2f}')

        except Exception as e:
            self.get_logger().warn(f'Cannot get transform {self.global_frame} -> {self.robot_frame}: {e}')

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'CSV saved: {self.csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PosePathLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
