import heapq
import math
import copy
import random

#小规模问题上性能比传统分支定界更强,但大规模问题上欠佳
class AdaptiveNode:
    def __init__(self, path, cost, bound, reduced_matrix, iteration):
        self.path = path          # 访问路径
        self.cost = cost          # 实际成本
        self.bound = bound        # 当前下界
        self.reduced_matrix = reduced_matrix  # 归约后的矩阵
        self.iteration = iteration  # 节点创建时的迭代次数
        
    def __lt__(self, other):
        return self.bound < other.bound

def adaptive_reduce_matrix(matrix, iteration, max_iterations=1000):
    """ 自适应矩阵归约策略 """
    n = len(matrix)
    reduction = 0
    decay = min(iteration / max_iterations, 1.0)  # 衰减系数
    
    # 深拷贝矩阵避免修改原始数据
    mat = copy.deepcopy(matrix)
    
    # ========== 行归约 ==========
    for i in range(n):
        min_val = min(mat[i])
        if min_val != math.inf:
            # 衰减归约量：后期减少归约强度
            actual_reduction = min_val * (1 - decay)
            reduction += actual_reduction
            mat[i] = [x - actual_reduction for x in mat[i]]
    
    # ========== 列归约 ==========
    if decay < 0.8:  # 衰减系数较大时跳过列归约
        for j in range(n):
            col = [mat[i][j] for i in range(n)]
            min_val = min(col)
            if min_val != math.inf:
                actual_reduction = min_val * (1 - decay)
                reduction += actual_reduction
                for i in range(n):
                    mat[i][j] -= actual_reduction
    
    return mat, reduction

def adaptive_tsp(original_matrix, max_iterations=1000):
    n = len(original_matrix)
    if n == 0:
        return 0, []
    
    heap = []
    min_cost = math.inf
    best_path = []
    global_iteration = 0  # 全局迭代计数器
    
    # 初始化
    initial_matrix = copy.deepcopy(original_matrix)
    reduced_matrix, reduction = adaptive_reduce_matrix(initial_matrix, 0, max_iterations)
    heapq.heappush(heap, AdaptiveNode(
        path=[0],
        cost=0,
        bound=reduction,
        reduced_matrix=reduced_matrix,
        iteration=0
    ))
    
    while heap:
        current = heapq.heappop(heap)
        global_iteration += 1
        
        # 剪枝条件
        if current.bound >= min_cost:
            continue
        
        # 完整路径检查
        if len(current.path) == n:
            final_cost = current.cost + original_matrix[current.path[-1]][0]
            if final_cost < min_cost:
                min_cost = final_cost
                best_path = current.path + [0]
            continue
        
        # 动态调整归约策略
        new_matrix, new_reduction = adaptive_reduce_matrix(
            current.reduced_matrix,
            current.iteration,
            max_iterations
        )
        
        # 扩展子节点
        from_city = current.path[-1]
        for to_city in range(n):
            if to_city not in current.path:
                step_cost = original_matrix[from_city][to_city]
                
                # 创建新节点的归约矩阵
                updated_matrix = copy.deepcopy(new_matrix)
                for i in range(n):
                    updated_matrix[from_city][i] = math.inf
                    updated_matrix[i][to_city] = math.inf
                if len(current.path) + 1 != n:
                    updated_matrix[to_city][0] = math.inf
                
                # 计算新下界
                reduced_mat, reduction = adaptive_reduce_matrix(
                    updated_matrix,
                    global_iteration,
                    max_iterations
                )
                new_bound = current.cost + step_cost + reduction
                
                if new_bound < min_cost:
                    new_node = AdaptiveNode(
                        path=current.path + [to_city],
                        cost=current.cost + step_cost,
                        bound=new_bound,
                        reduced_matrix=reduced_mat,
                        iteration=global_iteration
                    )
                    heapq.heappush(heap, new_node)
    
    return min_cost, best_path

# 测试验证
if __name__ == "__main__":
    # 4城市标准测试
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    print("=== 4城市测试 ===")
    cost, path = adaptive_tsp(example_matrix)
    print(f"最短路径长度: {cost}")  # 应输出80
    print(f"路径顺序: {path}")     # 应输出[0, 1, 3, 2, 0]
    
    # 5城市测试
    example_matrix_5 = [
        [0, 3, 1, 5, 8],
        [3, 0, 6, 7, 9],
        [1, 6, 0, 4, 2],
        [5, 7, 4, 0, 3],
        [8, 9, 2, 3, 0]
    ]
    
    print("\n=== 5城市测试 ===")
    cost, path = adaptive_tsp(example_matrix_5)
    print(f"最短路径长度: {cost}")  # 应输出16
    print(f"路径顺序: {path}")     # 可能路径: [0, 2, 4, 3, 1, 0]