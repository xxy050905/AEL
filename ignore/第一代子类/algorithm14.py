import math
import random
import numpy as np


#容易陷入局部最优,结果不准确,输出正确答案的概率不高,考虑弃用
def probabilistic_greedy_tsp(distance_matrix, initial_temp=1000, cooling_rate=0.95, max_iter=1000):
    """
    混合启发式TSP求解器（贪心+模拟退火）
    :param distance_matrix: 距离矩阵
    :param initial_temp: 初始温度
    :param cooling_rate: 冷却速率(0.9~0.999)
    :param max_iter: 最大迭代次数
    :return: (最短路径长度, 路径顺序)
    """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []
    
    best_cost = math.inf
    best_path = []
    
    # 多起点并行优化
    for start in range(n):
        current_temp = initial_temp
        path = [start]
        visited = set([start])
        current_cost = 0
        
        for _ in range(max_iter):
            if len(path) == n:
                # 闭合路径
                current_cost += distance_matrix[path[-1]][start]
                path.append(start)
                break
            
            current_city = path[-1]
            neighbors = []
            
            # 收集未访问邻居信息
            for city in range(n):
                if city not in visited:
                    delta_cost = distance_matrix[current_city][city]
                    neighbors.append((city, delta_cost))
            
            if not neighbors:
                break
            
            # 生成概率分布
            probabilities = []
            total = 0.0
            for city, cost in neighbors:
                # LLM生成的适应度公式：p = exp(-cost/(current_temp + 1e-7))
                prob = math.exp(-cost / (current_temp + 1e-7))  # 防止除以0
                probabilities.append(prob)
                total += prob
            
            # 归一化
            probabilities = [p/total for p in probabilities]
            
            # 依概率选择下一个城市
            chosen_idx = np.random.choice(len(neighbors), p=probabilities)
            next_city, step_cost = neighbors[chosen_idx]
            
            # 更新状态
            path.append(next_city)
            visited.add(next_city)
            current_cost += step_cost
            
            # 降温
            current_temp *= cooling_rate
        
        # 闭合路径检查
        if len(path) == n+1 and path[0] == path[-1]:
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path
    
    return best_cost, best_path

# 测试验证
if __name__ == "__main__":
    # 4城市测试（已知最优解80）
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cost, path = probabilistic_greedy_tsp(example_matrix, initial_temp=500, cooling_rate=0.99, max_iter=1000)
    print("=== 4城市测试 ===")
    print(f"最短路径长度: {cost}")  # 应输出80
    print(f"路径顺序: {path}")     # 可能路径: [0,1,3,2,0]

    # 5城市测试（已知最优解16）
    example_matrix_5 = [
        [0, 3, 1, 5, 8],
        [3, 0, 6, 7, 9],
        [1, 6, 0, 4, 2],
        [5, 7, 4, 0, 3],
        [8, 9, 2, 3, 0]
    ]
    cost, path = probabilistic_greedy_tsp(example_matrix_5, initial_temp=1000, cooling_rate=0.95, max_iter=2000)
    print("\n=== 5城市测试 ===")
    print(f"最短路径长度: {cost}")  # 应输出16
    print(f"路径顺序: {path}")     # 可能路径: [0,2,4,3,1,0]