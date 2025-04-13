import heapq
import math
import copy
import itertools

MAX_CITIES = 20

#分支界定法与动态规划结合
"""
算法流程总结
预处理阶段 :通过Held-Karp算法预计算所有子集的最短路径成本(dp_table)。
分支定界阶段 :
使用优先队列扩展节点，每次选择下界最小的节点。
结合动态规划的预计算结果快速计算剩余路径的下界，剪枝无效分支。
路径解码阶段 ：根据最终的最优掩码和动态规划表重建路径
"""
class EnhancedNode:
    def __init__(self, mask, cost, bound, reduced_matrix, current_city):
        self.mask = mask
        self.cost = cost
        self.bound = bound
        self.reduced_matrix = reduced_matrix
        self.current_city = current_city

    def __lt__(self, other):
        return self.bound < other.bound

def reduce_matrix(matrix):
    n = len(matrix)
    reduction = 0
    # 行归约
    for i in range(n):
        min_val = min(matrix[i])
        if min_val != math.inf:
            reduction += min_val
            matrix[i] = [x - min_val for x in matrix[i]]
    
    # 列归约（修复索引错误）
    for j in range(n):
        col_values = [matrix[i][j] for i in range(n)]
        min_val = min(col_values)
        if min_val != math.inf:
            reduction += min_val
            for i in range(n):
                matrix[i][j] -= min_val
    
    return matrix, reduction

def held_karp_precompute(matrix):
    """ 修正的动态规划预计算 """
    n = len(matrix)
    dp = {}
    
    # 城市编号从1开始处理（对应二进制位0~n-2）
    for v in range(1, n):
        mask = 1 << (v-1)
        dp[(v, mask)] = matrix[0][v]

    for subset_size in range(2, n):
        for cities in itertools.combinations(range(1, n), subset_size):
            mask = sum(1 << (c-1) for c in cities)
            for v in cities:
                prev_mask = mask ^ (1 << (v-1))
                min_cost = min(
                    dp.get((u, prev_mask), math.inf) + matrix[u][v]
                    for u in cities if u != v
                )
                if min_cost != math.inf:
                    dp[(v, mask)] = min_cost
    return dp

def decode_path(best_mask, dp_table, n):
    """ 精确路径解码函数 """
    path = [0]
    current_city = 0
    visited = 1 << 0  # 已访问掩码
    
    for _ in range(n-1):
        next_city = None
        min_cost = math.inf
        
        # 遍历所有可能的下一个城市（排除起点）
        for city in range(1, n):
            if not (visited & (1 << city)):
                # 计算新的已访问掩码
                new_visited = visited | (1 << city)
                # 提取非0城市的已访问位作为动态规划掩码
                dp_mask = (new_visited >> 1) & ((1 << (n-1)) - 1)
                
                # 查询动态规划表
                key = (city, dp_mask)
                cost = dp_table.get(key, math.inf)
                
                if cost < min_cost:
                    min_cost = cost
                    next_city = city
        
        if next_city is None:
            break
        
        path.append(next_city)
        visited |= 1 << next_city
        current_city = next_city
    
    path.append(0)
    return path

def hybrid_tsp(original_matrix):
    n = len(original_matrix)
    if n == 0:
        return 0, []
    if n > MAX_CITIES:
        raise ValueError(f"城市数量超过限制{MAX_CITIES}")

    dp_table = held_karp_precompute(original_matrix)
    heap = []
    min_cost = math.inf
    best_mask = 0

    # 初始化根节点
    initial_matrix = [row.copy() for row in original_matrix]
    reduced_matrix, reduction = reduce_matrix(initial_matrix)
    heapq.heappush(heap, EnhancedNode(
        mask=1 << 0,
        cost=0,
        bound=reduction,
        reduced_matrix=reduced_matrix,
        current_city=0
    ))

    while heap:
        current = heapq.heappop(heap)

        # 剪枝条件（关键修复点）
        remaining_mask = ((1 << n) - 1) ^ current.mask
        if remaining_mask == 0:
            final_cost = current.cost + original_matrix[current.current_city][0]
            if final_cost < min_cost:
                min_cost = final_cost
                best_mask = current.mask
            continue
        
        # 动态规划下界计算
        dp_remaining_mask = (remaining_mask >> 1) & ((1 << (n-1)) - 1)
        dp_lower = 0
        if current.current_city != 0:
            key = (current.current_city, dp_remaining_mask)
            dp_lower = dp_table.get(key, 0)
        
        if current.bound + dp_lower >= min_cost:
            continue

        # 扩展子节点
        for to_city in range(n):
            if not (current.mask & (1 << to_city)):
                step_cost = original_matrix[current.current_city][to_city]
                if step_cost == math.inf:
                    continue

                # 创建新节点
                new_mask = current.mask | (1 << to_city)
                new_matrix = copy.deepcopy(current.reduced_matrix)
                
                # 更新禁止访问的边
                for i in range(n):
                    new_matrix[current.current_city][i] = math.inf
                    new_matrix[i][to_city] = math.inf
                if bin(new_mask).count('1') < n:
                    new_matrix[to_city][0] = math.inf
                
                # 计算新下界
                new_reduced, reduction = reduce_matrix(new_matrix)
                remaining_new_mask = ((1 << n) - 1) ^ new_mask
                dp_remaining = dp_table.get((to_city, (remaining_new_mask >> 1)), 0)
                new_bound = current.cost + step_cost + reduction + dp_remaining

                if new_bound < min_cost:
                    heapq.heappush(heap, EnhancedNode(
                        mask=new_mask,
                        cost=current.cost + step_cost,
                        bound=new_bound,
                        reduced_matrix=new_reduced,
                        current_city=to_city
                    ))

    # 最终路径解码
    path = decode_path(best_mask, dp_table, n)
    return (min_cost, path) if min_cost != math.inf else (float('inf'), [])

# 验证测试
if __name__ == "__main__":
    # 4城市标准测试
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    cost, path = hybrid_tsp(example_matrix)
    print(f"最短路径长度: {cost}")  # 正确输出80
    print(f"路径顺序: {path}")     # 正确输出[0, 1, 3, 2, 0]