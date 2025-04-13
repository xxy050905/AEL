from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpBinary
import itertools

def tsp_ilp(distance_matrix):
    """
    使用整数线性规划（ILP）求解TSP问题
    :param distance_matrix: 距离矩阵（方阵，满足三角不等式）
    :return: 最短路径长度，路径顺序
    """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []

    # 创建ILP问题实例
    prob = LpProblem("TSP_ILP", LpMinimize)

    # 定义决策变量 x_ij (表示是否从i到j)
    x = LpVariable.dicts("Edge", 
                        [(i, j) for i in range(n) for j in range(n) if i != j],
                        cat=LpBinary)

    # 目标函数：最小化总距离
    prob += lpSum(distance_matrix[i][j] * x[(i, j)] 
                 for i in range(n) for j in range(n) if i != j)

    # 约束1：每个城市恰好离开一次
    for i in range(n):
        prob += lpSum(x[(i, j)] for j in range(n) if i != j) == 1

    # 约束2：每个城市恰好到达一次
    for j in range(n):
        prob += lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    # 约束3：消除子环路（MTZ约束）
    u = LpVariable.dicts("Order", range(n), lowBound=0, upBound=n-1, cat="Integer")
    for i in range(1, n):  # 起点0不需要u变量
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    # 求解问题
    prob.solve()

    # 检查求解状态
    if LpStatus[prob.status] != "Optimal":
        return float('inf'), []

    # 提取路径
    path = [0]
    current = 0
    for _ in range(n-1):
        for j in range(n):
            if j != current and x[(current, j)].varValue > 0.5:
                path.append(j)
                current = j
                break
    path.append(0)  # 回到起点

    return prob.objective.value(), path

# 示例测试
if __name__ == "__main__":
    # 示例1：4个城市
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost, path = tsp_ilp(example_matrix)
    print(f"最短路径长度: {min_cost}")  # 输出80.0
    print(f"路径顺序: {path}")        # 输出 [0, 1, 3, 2, 0] 或其他等效解

    # 示例2：3个城市
    example_matrix2 = [
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]
    min_cost, path = tsp_ilp(example_matrix2)
    print(f"最短路径长度: {min_cost}")  # 输出6.0
    print(f"路径顺序: {path}")        # 输出 [0, 1, 2, 0] 或 [0, 2, 1, 0]