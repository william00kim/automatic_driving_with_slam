import numpy as np
import cv2
from geometry_msgs.msg import Point, PoseStamped

class FrontierSearch:
    def __init__(self, potential_scale=1e-3, gain_scale=1.0, min_frontier_size=0.3):
        self.potential_scale = potential_scale
        self.gain_scale = gain_scale
        self.min_frontier_size = min_frontier_size 

    def search_from(self, map_msg, robot_pos):
        """
        map_msg: OccupancyGrid 메시지 전체 (info의 origin 정보 활용을 위함)
        robot_pos: (x, y) 실시간 로봇 위치 (World 기준)
        """
        width = map_msg.info.width
        height = map_msg.info.height
        res = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        # 1. 넘파이 변환 및 마스킹 (속도 최적화)
        grid = np.array(map_msg.data).reshape((height, width))
        
        # Unknown(-1)은 255, Free(0)는 0으로 변환하여 이진 영상 생성
        unknown_mask = (grid == -1).astype(np.uint8) * 255
        free_mask = (grid == 0).astype(np.uint8) * 255
        
        # 2. Frontier 검출: Free 영역을 팽창시켜 Unknown과 겹치는 선 추출
        kernel = np.ones((3, 3), np.uint8)
        dilated_free = cv2.dilate(free_mask, kernel)
        frontier_mask = cv2.bitwise_and(unknown_mask, dilated_free)

        # 3. 군집화
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask)

        frontier_list = []


        for i in range(1, num_labels):
            size_in_pixels = stats[i, cv2.CC_STAT_AREA]
            # 픽셀 개수가 아닌 실제 길이(m) 근사치로 계산
            if size_in_pixels * res < self.min_frontier_size:
                continue

            # 픽셀 좌표 (cx, cy)
            cx, cy = centroids[i]
            
            # 4. 좌표 변환: Pixel -> World (중요!)
            # 배열의 index는 (row, col)이므로 cy가 행(y), cx가 열(x)입니다.
            world_x = (cx * res) + origin_x
            world_y = (cy * res) + origin_y
            
            # 로봇과의 거리 계산
            dist = np.sqrt((world_x - robot_pos[0])**2 + (world_y - robot_pos[1])**2)
            
            frontier = {
                'world_point': (world_x, world_y),
                'cost': (self.potential_scale * dist) - (self.gain_scale * size_in_pixels * res)
            }
            
            frontier_list.append(frontier)

        # Cost 낮은 순 정렬
        frontier_list.sort(key=lambda x: x['cost'])
        
        print(frontier_list)
        
        return frontier_list