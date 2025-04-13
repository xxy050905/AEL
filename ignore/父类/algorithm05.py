import itertools
#动态规划
def held_karp_tsp(distance_matrix):

    n = len(distance_matrix)
    if n == 0:
        return 0  # 空输入直接返回0
    
    # 初始化动态规划表，键为元组（当前城市，访问过的城市集合）
    dp = {}
    
    # 步骤1：初始化单城市子集（从起点0出发到其他城市）
    for v in range(1, n):
        # 二进制掩码表示仅包含当前城市v的子集（例如：城市1对应掩码0b1）
        mask = 1 << (v - 1)
        dp[(v, mask)] = distance_matrix[0][v]
    
    # 步骤2：逐步构建更大的子集（从大小为2到n-1的子集）
    for subset_size in range(2, n):
        # 生成所有包含subset_size个城市的组合（不包含起点0）
        for cities in itertools.combinations(range(1, n), subset_size):
            # 计算当前子集的二进制掩码（例如：[1,2]对应掩码0b11）
            mask = 0
            for city in cities:
                mask |= 1 << (city - 1)
            
            # 对子集中的每个城市v，计算到达该城市的最短路径
            for v in cities:
                # 从当前子集移除v得到前一个子集的掩码
                prev_mask = mask ^ (1 << (v - 1))
                min_dist = float('inf')
                
                # 遍历前一个子集中的所有可能的上一个城市u
                for u in cities:
                    if u == v:
                        continue  # 跳过当前城市v本身
                    # 如果(u, prev_mask)的状态已存在，则更新最短距离
                    if (u, prev_mask) in dp:
                        current_dist = dp[(u, prev_mask)] + distance_matrix[u][v]
                        if current_dist < min_dist:
                            min_dist = current_dist
                
                # 保存当前状态的最短距离
                if min_dist != float('inf'):
                    dp[(v, mask)] = min_dist

    # 步骤3：计算最终结果（回到起点0）
    full_mask = (1 << (n - 1)) - 1  # 所有城市都被访问过的掩码（例如：n=4对应0b111）
    min_total = float('inf')
    
    # 遍历所有可能的最后访问城市，加上返回起点的距离
    for v in range(1, n):
        if (v, full_mask) in dp:
            total = dp[(v, full_mask)] + distance_matrix[v][0]
            if total < min_total:
                min_total = total
    
    return min_total if min_total != float('inf') else -1
