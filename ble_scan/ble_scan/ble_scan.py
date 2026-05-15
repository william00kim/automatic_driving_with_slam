import os
import csv
import serial
from datetime import datetime

import rclpy
from rclpy.node import Node


class BLE_SCAN(Node):
    def __init__(self):
        super().__init__('ble_scan')
        self.get_logger().info('BLE Scan node has been started.')

        

        self.TARGET_MAC = [
            "C8:3E:03:09:F9:B9",
            "DF:DF:0C:A3:80:F9",
            "E9:88:6E:C6:B6:0E",
            "E6:4F:B3:54:95:51",
        ]

        self.TARGET_MAC = [mac.lower() for mac in self.TARGET_MAC]

        self.get_logger().info(f'target MACs: {self.TARGET_MAC}')

        self.SERIAL_PORT = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_ECDA3B5527E4-if01"
        self.BAUDRATE = 115200
        self.LOG_DIR = os.path.expanduser("~/ble_scan_logs")
        self.WINDOW_MS = 500
        self.RSSI_MISSING = -100

        self.buffer = {}
        self.window_start_ms = 0.0

        self.csv_file = None
        self.writer = None
        self.current_csv_path = None

        os.makedirs(self.LOG_DIR, exist_ok=True)

        self.ser = serial.Serial(
            self.SERIAL_PORT,
            self.BAUDRATE,
            timeout=0.05
        )

        self.create_new_csv_file()

        self.read_timer = self.create_timer(0.1, self.read_serial)
        self.flush_timer = self.create_timer(0.1, self.check_flush_window)

    def now_ms(self):
        return self.get_clock().now().nanoseconds / 1_000_000.0

    def create_new_csv_file(self):
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_csv_path = os.path.join(
            self.LOG_DIR,
            f"ble_matrix_{now_str}.csv"
        )

        self.csv_file = open(
            self.current_csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["ros_timestamp_ms"] + self.TARGET_MAC)
        self.csv_file.flush()

        self.window_start_ms = self.now_ms()

        self.get_logger().info(f"CSV created: {self.current_csv_path}")
        self.get_logger().info(f"ROS start time: {self.window_start_ms:.3f} ms")

    def add_sample(self, address, rssi):

        if address not in self.TARGET_MAC:
            return

        self.buffer.setdefault(address, []).append(rssi)

    def read_serial(self):
        try:
            if not self.ser.is_open:
                return

            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if not line:
                    continue

                if line.startswith("READY") or line.startswith("nano_time_ms"):
                    continue

                parts = [p.strip() for p in line.split(",")]

                if len(parts) < 5:
                    continue

                if parts[0] != "BLE":
                    continue

                address = parts[2].lower()

                try:
                    rssi = int(parts[3])
                except ValueError:
                    continue

                self.add_sample(address, rssi)

        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

    def check_flush_window(self):
        if self.writer is None:
            return

        current_ms = self.now_ms()

        while current_ms >= self.window_start_ms + self.WINDOW_MS:
            self.flush_window()

    def flush_window(self):
        row = [f"{self.window_start_ms:.3f}"]

        for mac in self.TARGET_MAC:
            values = self.buffer.get(mac, [])

            if values:
                sorted_values = sorted(values)
                median_rssi = sorted_values[len(sorted_values) // 2]
            else:
                median_rssi = self.RSSI_MISSING

            row.append(median_rssi)

        self.writer.writerow(row)
        self.csv_file.flush()

        self.get_logger().info(f"[FLUSH] {row}")

        self.buffer = {}
        self.window_start_ms += self.WINDOW_MS

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()

        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()

        self.get_logger().info("BLE Scan node closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BLE_SCAN()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
