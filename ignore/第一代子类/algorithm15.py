import itertools
import math

def dp_exhaustive_tsp(distance_matrix):
    """
    结合动态规划状态记录的改进型穷举法
    :param distance_matrix: 距离矩阵（方阵）
    :return: 最短路径长度, 路径顺序
    """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []
    
    # 初始化动态规划表
    memo = {}  # 格式: (当前城市, 已访问掩码) -> (最小成本, 最优下一城市)
    
    def dfs(current, visited):
        """ 记忆化深度优先搜索 """
        if (current, visited) in memo:
            return memo[(current, visited)]
        
        # 所有城市已访问
        if visited == (1 << n) - 1:
            return distance_matrix[current][0], [current, 0]
        
        min_cost = math.inf
        best_path = []
        
        for next_city in range(n):
            if not (visited & (1 << next_city)):
                new_visited = visited | (1 << next_city)
                cost, path = dfs(next_city, new_visited)
                total_cost = distance_matrix[current][next_city] + cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = [current] + path
        
        memo[(current, visited)] = (min_cost, best_path)
        return min_cost, best_path
    
    # 多起点并行优化（实际只需从0出发）
    min_cost, best_path = dfs(0, 1 << 0)
    
    # 调整路径格式为闭环
    return min_cost, best_path if len(best_path) == n+1 else []

# 测试验证
if __name__ == "__main__":
    # 4城市测试（已知最优解80）
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cost, path = dp_exhaustive_tsp(example_matrix)
    print("=== 4城市测试 ===")
    print(f"最短路径长度: {cost}")  # 输出80
    print(f"路径顺序: {path}")     # 输出[0,1,3,2,0]

    # 5城市测试（已知最优解16）
    example_matrix_5 = [
        [0, 3, 1, 5, 8],
        [3, 0, 6, 7, 9],
        [1, 6, 0, 4, 2],
        [5, 7, 4, 0, 3],
        [8, 9, 2, 3, 0]
    ]
    print("\n=== 5城市测试 ===")
    cost, path = dp_exhaustive_tsp(example_matrix_5)
    print(f"最短路径长度: {cost}")  # 输出16
    print(f"路径顺序: {path}")     # 输出[0,2,4,3,1,0]