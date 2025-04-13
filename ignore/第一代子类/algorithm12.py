from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpBinary
import itertools

def greedy_nearest_neighbor(matrix):
    """ 改进的贪心算法，返回路径边集合 """
    n = len(matrix)
    min_total = float('inf')
    best_path = []
    
    for start in range(n):
        visited = [False]*n
        path = [start]
        visited[start] = True
        total = 0
        current = start
        
        for _ in range(n-1):
            min_dist = float('inf')
            next_city = -1
            for neighbor in range(n):
                if not visited[neighbor] and matrix[current][neighbor] < min_dist:
                    min_dist = matrix[current][neighbor]
                    next_city = neighbor
            if next_city == -1:
                break
            path.append(next_city)
            visited[next_city] = True
            total += min_dist
            current = next_city
        
        if len(path) == n:
            total += matrix[path[-1]][start]
            if total < min_total:
                min_total = total
                best_path = path + [start]
    
    # 提取路径边集合
    edges = set()
    for i in range(len(best_path)-1):
        u = best_path[i]
        v = best_path[i+1]
        edges.add((u, v))
    return min_total, best_path, edges

def tsp_greedy_ilp(distance_matrix, neighborhood_radius=2):
    """ 贪心引导的约束规划算法 """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []
    
    # 阶段1：生成贪心初始解及候选边
    greedy_cost, greedy_path, greedy_edges = greedy_nearest_neighbor(distance_matrix)
    
    # 生成候选边集合（贪心路径边 + 邻域边）
    candidate_edges = set(greedy_edges)
    for u in range(n):
        # 获取每个城市最近的k个邻居
        neighbors = sorted([(distance_matrix[u][v], v) for v in range(n) if v != u])
        for i in range(min(neighborhood_radius, len(neighbors))):
            candidate_edges.add((u, neighbors[i][1]))
    
    # 阶段2：构建ILP模型
    prob = LpProblem("TSP_Greedy_ILP", LpMinimize)
    
    # 定义变量（仅候选边）
    x = LpVariable.dicts("Edge", 
                        [(i, j) for i, j in candidate_edges if i != j],
                        cat=LpBinary)
    
    # 目标函数
    prob += lpSum(distance_matrix[i][j] * x[(i, j)] for (i, j) in x)
    
    # 约束1：每个城市离开一次
    for i in range(n):
        prob += lpSum(x[(i, j)] for j in range(n) if (i, j) in x) == 1
    
    # 约束2：每个城市到达一次
    for j in range(n):
        prob += lpSum(x[(i, j)] for i in range(n) if (i, j) in x) == 1
    
    # 约束3：MTZ子环路消除（仅候选边）
    u = LpVariable.dicts("Order", range(n), lowBound=0, upBound=n-1, cat="Integer")
    for (i, j) in x:
        if i != 0 and j != 0:
            prob += u[i] - u[j] + n * x[(i, j)] <= n - 1
    
    # 阶段3：设置初始解
    for (i, j) in x:
        if (i, j) in greedy_edges:
            x[(i, j)].setInitialValue(1.0)
        else:
            x[(i, j)].setInitialValue(0.0)
    
    # 求解（需支持初始解的求解器如CPLEX或Gurobi）
    try:
        prob.solve()
    except:
        prob.solver = None
        prob.solve()
    
    # 验证解
    if LpStatus[prob.status] != "Optimal":
        return greedy_cost, greedy_path
    
    # 提取路径
    path = [0]
    current = 0
    for _ in range(n):
        for j in range(n):
            if current != j and (current, j) in x and x[(current, j)].varValue > 0.5:
                path.append(j)
                current = j
                break
        if current == 0:
            break
    path.append(0)
    
    # 计算实际成本
    ilp_cost = sum(distance_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
    return min(ilp_cost, greedy_cost), path if ilp_cost < greedy_cost else greedy_path

# 测试验证
if __name__ == "__main__":
    # 4城市测试
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    min_cost, path = tsp_greedy_ilp(example_matrix)
    print(f"最短路径长度: {min_cost}")  # 应输出80
    print(f"路径顺序: {path}")        # 应输出[0, 1, 3, 2, 0]