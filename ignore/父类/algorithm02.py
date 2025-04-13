import itertools
#穷举法
def exhaustive_tsp(distance_matrix):

    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []  # 处理0或1个城市的边界情况

    min_distance = float('inf')
    best_path = []

    # 固定起点为0，生成剩余城市的全排列（减少一半重复计算）
    for permutation in itertools.permutations(range(1, n)):
        current_path = [0] + list(permutation) + [0]  # 添加起点和终点形成环路
        total_distance = 0

        # 计算环路总距离
        for i in range(len(current_path)-1):
            from_city = current_path[i]
            to_city = current_path[i+1]
            total_distance += distance_matrix[from_city][to_city]

        # 更新最优解
        if total_distance < min_distance:
            min_distance = total_distance
            best_path = current_path

    return min_distance, best_path

# 示例测试
if __name__ == "__main__":
    # 示例1：4个城市（与分支定界法相同输入）
    example_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_dist, path = exhaustive_tsp(example_matrix)
    print(f"最短路径长度: {min_dist}")  # 输出80
    print(f"路径顺序: {path}")        # 输出[0, 1, 3, 2, 0] 或其他等效排列

    # 示例2：3个城市
    example_matrix2 = [
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]
    min_dist, path = exhaustive_tsp(example_matrix2)
    print(f"最短路径长度: {min_dist}")  # 输出6（0->1->2->0 或 0->2->1->0）
    print(f"路径顺序: {path}")