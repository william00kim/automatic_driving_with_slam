import numpy as np
from collections import deque
from geometry_msgs.msg import Point, PoseStamped


class FrontierSearch:
    def __init__(self, potential_scale=1e-3, gain_scale=1.0, min_frontier_size=0.05):
        self.potential_scale = potential_scale
        self.gain_scale = gain_scale
        self.min_frontier_size = min_frontier_size 

    # -------------------------------
    # dilation (cv2.dilate 대체)
    # -------------------------------
    def dilate(self, mask):
        h, w = mask.shape
        result = np.zeros_like(mask, dtype=bool)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))

                # 경계 wrap 제거 (중요)
                if dy == -1:
                    shifted[-1, :] = 0
                if dy == 1:
                    shifted[0, :] = 0
                if dx == -1:
                    shifted[:, -1] = 0
                if dx == 1:
                    shifted[:, 0] = 0

                result = np.logical_or(result, shifted)

        return result

    # -------------------------------
    # connected components (BFS)
    # -------------------------------
    def get_clusters_BFS(self, mask):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    queue = deque([(y, x)])
                    visited[y, x] = True
                    cluster = []

                    while queue:
                        cy, cx = queue.popleft()
                        cluster.append((cy, cx))

                        for dy, dx in directions:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))

                    clusters.append(cluster)

        return clusters
    
    # -------------------------------
    # connected components (DFS)
    # -------------------------------
    def get_clusters_DFS(self, mask):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    stack = [(y, x)]
                    visited[y, x] = True
                    cluster = []

                    while stack:
                        cy, cx = stack.pop()
                        cluster.append((cy, cx))

                        for dy, dx in directions:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))

                    clusters.append(cluster)

        return clusters

    # -------------------------------
    # centroid 계산
    # -------------------------------
    def get_centroid(self, cluster):
        ys = [p[0] for p in cluster]
        xs = [p[1] for p in cluster]
        return np.mean(xs), np.mean(ys)

    # -------------------------------
    # 메인 함수
    # -------------------------------
    def search_from(self, map_msg, robot_pos):

        width = map_msg.info.width
        height = map_msg.info.height
        res = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        # 1. grid 변환
        grid = np.array(map_msg.data).reshape((height, width))

        # 2. mask 생성 (bool로 처리)
        unknown_mask = (grid == -1)
        free_mask = (grid == 0)

        # 3. dilation
        dilated_free = self.dilate(free_mask)

        # 4. frontier 추출
        frontier_mask = np.logical_and(unknown_mask, dilated_free)

        # 5. 클러스터링
        clusters = self.get_clusters_DFS(frontier_mask)

        frontier_list = []

        for cluster in clusters:
            size_in_pixels = len(cluster)

            if size_in_pixels * res < self.min_frontier_size:
                continue

            cx, cy = self.get_centroid(cluster)

            # Pixel → World
            world_x = (cx * res) + origin_x
            world_y = (cy * res) + origin_y

            dist = np.sqrt((world_x - robot_pos[0])**2 + (world_y - robot_pos[1])**2)

            frontier = {
                'world_point': (world_x, world_y),
                'cost': (self.potential_scale * dist) - (self.gain_scale * size_in_pixels * res)
            }

            frontier_list.append(frontier)

        frontier_list.sort(key=lambda x: x['cost'])

        return frontier_list