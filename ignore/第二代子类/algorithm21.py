# cSpell:ignore Karp 
import heapq
import math
import copy
import itertools
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpStatus
from concurrent.futures import ThreadPoolExecutor
# 新增导入语句（修复未定义变量）
from algorithm12 import tsp_greedy_ilp  # 导入algorithm2的ILP算法
from algorithm13 import adaptive_tsp    # 导入algorithm3的自适应算法
from algorithm11 import hybrid_tsp      # 导入自身的核心算法

MAX_CITIES = 20

# 增强节点类（交叉algorithm1和algorithm3）
class HybridAdaptiveNode:
    def __init__(self, mask, cost, bound, reduced_matrix, current_city, iteration):
        self.mask = mask
        self.cost = cost
        self.bound = bound
        self.reduced_matrix = reduced_matrix  # 交叉algorithm1的归约矩阵
        self.current_city = current_city
        self.iteration = iteration           # 变异自algorithm3的迭代感知
    
    def __lt__(self, other):
        return self.bound < other.bound

# 自适应矩阵归约（变异algorithm3的衰减策略）
def adaptive_reduce_matrix(matrix, iteration, max_iters=1000):
    n = len(matrix)
    decay = (iteration / max_iters) ** 2
    mat = copy.deepcopy(matrix)
    reduction = 0.0  # 初始化归约量
    
    # 行归约（累计归约量）
    for i in range(n):
        min_val = min(mat[i])
        if min_val != math.inf:
            row_red = min_val * (1 - decay)
            reduction += row_red
            mat[i] = [x - row_red for x in mat[i]]
    
    # 条件列归约（累计归约量）
    if decay < 0.7:
        for j in range(n):
            col_min = min(mat[i][j] for i in range(n))
            if col_min != math.inf:
                col_red = col_min * (1 - decay/2)
                reduction += col_red
                for i in range(n):
                    mat[i][j] -= col_red
    
    return mat, reduction  # 正确返回两个值

# 交叉algorithm1的Held-Karp与algorithm2的贪心策略
def hybrid_precompute(matrix):
    n = len(matrix)
    dp_table = {}
    
    # 贪心初始化（变异点）
    greedy_path = []
    current = 0
    visited = {0}
    while len(greedy_path) < n:
        greedy_path.append(current)
        neighbors = sorted(
            (matrix[current][j], j) 
            for j in range(n) if j not in visited
        )
        if not neighbors: break
        current = neighbors[0][1]
        visited.add(current)
    
    # 动态规划预计算（交叉策略）
    for v in range(1, n):
        mask = 1 << (v-1)
        dp_table[(v, mask)] = matrix[0][v]
    
    for subset_size in range(2, n):
        for cities in itertools.combinations(range(1, n), subset_size):
            mask = sum(1 << (c-1) for c in cities)
            for v in cities:
                prev_mask = mask ^ (1 << (v-1))
                min_cost = min(
                    dp_table.get((u, prev_mask), math.inf) + matrix[u][v]
                    for u in cities if u != v
                )
                dp_table[(v, mask)] = min_cost
    
    return dp_table, greedy_path

# 第二代融合算法（核心）
def evolved_tsp(matrix):
    n = len(matrix)
    if n == 0:
        return (0, [])
    if n > MAX_CITIES:
        return tsp_greedy_ilp(matrix)  # 降级到纯ILP
    
    # 阶段1：混合预处理
    dp_table, greedy_path = hybrid_precompute(matrix)
    max_iters = 2**n
    heap = []
    
    # 初始节点（交叉algorithm1和algorithm3）
    initial_matrix = copy.deepcopy(matrix)
    reduced, reduction = adaptive_reduce_matrix(initial_matrix, 0, max_iters)
    heapq.heappush(heap, HybridAdaptiveNode(
        mask=1 << 0,
        cost=0,
        bound=reduction,
        reduced_matrix=reduced,
        current_city=0,
        iteration=0
    ))
    
    min_cost = math.inf
    best_mask = 0
    
    while heap:
        current = heapq.heappop(heap)
        remaining = ((1 << n) - 1) ^ current.mask
        
        # 终止条件
        if remaining == 0:
            final_cost = current.cost + matrix[current.current_city][0]
            if final_cost < min_cost:
                min_cost = final_cost
                best_mask = current.mask
            continue
        
        # 动态下界计算（交叉策略）
        dp_remaining_mask = (remaining >> 1) & ((1 << (n-1)) - 1)
        dp_lower = dp_table.get((current.current_city, dp_remaining_mask), 0)
        
        # 剪枝条件（增强版）
        if current.bound + dp_lower * (1 - current.iteration/max_iters) >= min_cost:
            continue
        
        # 扩展子节点
        for to_city in range(n):
            if not (current.mask & (1 << to_city)):
                step_cost = matrix[current.current_city][to_city]
                if step_cost == math.inf:
                    continue
                
                # 自适应矩阵归约（变异）
                new_matrix = copy.deepcopy(current.reduced_matrix)
                for i in range(n):
                    new_matrix[current.current_city][i] = math.inf
                    new_matrix[i][to_city] = math.inf
                if bin(current.mask | (1 << to_city)).count('1') < n:
                    new_matrix[to_city][0] = math.inf
                
                # 非线性衰减归约
                reduced_mat, reduction = adaptive_reduce_matrix(
                    new_matrix,
                    current.iteration,
                    max_iters
                )
                
                # 计算新下界（融合ILP约束思想）
                new_bound = current.cost + step_cost + reduction + dp_lower * 0.8
                
                if new_bound < min_cost:
                    heapq.heappush(heap, HybridAdaptiveNode(
                        mask=current.mask | (1 << to_city),
                        cost=current.cost + step_cost,
                        bound=new_bound,
                        reduced_matrix=reduced_mat,
                        current_city=to_city,
                        iteration=current.iteration + 1
                    ))
    
    # 解码路径（结合贪心路径）
    path = decode_path(best_mask, dp_table, n, greedy_path)
    return (min_cost, path) if min_cost != math.inf else (float('inf'), [])

# 改进的路径解码器（交叉贪心策略）
def decode_path(best_mask, dp_table, n, greedy_path):
    path = []
    current = 0
    visited = 1 << 0
    
    # 优先使用贪心路径
    for city in greedy_path:
        if (visited & (1 << city)) and city not in path:
            path.append(city)
    
    # 动态规划补全
    for _ in range(n - len(path)):
        next_city = None
        min_cost = math.inf
        for city in range(n):
            if not (visited & (1 << city)):
                dp_mask = (visited | (1 << city)) >> 1
                key = (city, dp_mask)
                cost = dp_table.get(key, math.inf)
                if cost < min_cost:
                    min_cost = cost
                    next_city = city
        if next_city is None:
            break
        path.append(next_city)
        visited |= 1 << next_city
    
    path.append(0)
    return path

# 并行化执行策略（新增变异）
def parallel_evolved_tsp(matrix):
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(evolved_tsp, matrix)
        future2 = executor.submit(tsp_greedy_ilp, matrix)
        future3 = executor.submit(adaptive_tsp, matrix)
        results = [f.result() for f in [future1, future2, future3]]
    return min(results, key=lambda x: x[0])

# 测试验证
if __name__ == "__main__":
    # 4城市标准测试
    test_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    # 执行算法
    print("=== 4城市测试 ===")
    cost, path = evolved_tsp(test_matrix)
    print(f"最短路径长度: {cost}")  # 应输出80
    print(f"路径顺序: {path}")      # 应输出[0, 1, 3, 2, 0]
    
    # 性能对比
    print("=== 性能对比 ===")
    algorithms = {
        "Algorithm1": hybrid_tsp,
        "Algorithm2": tsp_greedy_ilp,
        "Algorithm3": adaptive_tsp,
        "Evolved": evolved_tsp
    }
    
    for name, func in algorithms.items():
        cost, _ = func(test_matrix)
        print(f"{name:>10}: {cost}")