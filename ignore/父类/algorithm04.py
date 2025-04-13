import itertools
def greedy_nearest_neighbor_tsp(distance_matrix):
    """
    使用贪心最近邻法求解TSP问题
    :param distance_matrix: 距离矩阵（方阵，满足三角不等式）
    :return: 最短路径长度，路径顺序
    """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []
    
    min_total = float('inf')
    best_path = []
    
    # 遍历所有可能的起点（可选优化：固定起点时移除外层循环）
    for start in range(n):
        visited = [False] * n
        path = [start]
        visited[start] = True
        total = 0
        
        # 逐步访问最近邻节点
        current = start
        for _ in range(n-1):
            min_dist = float('inf')
            next_city = -1
            # 寻找最近的未访问城市
            for neighbor in range(n):
                if not visited[neighbor] and distance_matrix[current][neighbor] < min_dist:
                    min_dist = distance_matrix[current][neighbor]
                    next_city = neighbor
            # 更新路径和成本
            if next_city == -1:
                break  # 无未访问城市（异常情况）
            path.append(next_city)
            visited[next_city] = True
            total += min_dist
            current = next_city
        
        # 返回起点并更新最优解
        if len(path) == n:
            total += distance_matrix[path[-1]][start]
            if total < min_total:
                min_total = total
                best_path = path + [start]
    
    return min_total, best_path

# 示例测试
if __name__ == "__main__":
    # 示例1：4个城市（与标准测试一致）
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost, path = greedy_nearest_neighbor_tsp(example_matrix)
    print(f"最短路径长度: {min_cost}")  # 输出80（若起点0则路径0->1->3->2->0）
    print(f"路径顺序: {path}")

    # 示例2：5个城市
    example_matrix2 = [
        [0, 3, 1, 5, 8],
        [3, 0, 6, 7, 9],
        [1, 6, 0, 4, 2],
        [5, 7, 4, 0, 3],
        [8, 9, 2, 3, 0]
    ]
    min_cost, path = greedy_nearest_neighbor_tsp(example_matrix2)
    print(f"最短路径长度: {min_cost}")  # 输出16（路径可能为0->2->4->3->1->0）
    print(f"路径顺序: {path}")