import sys
import math
import random
import json
from collections import defaultdict

sys.path.append(r"D:\Paper\Algorithm Evolution Using Large Language Model\code\AEL")
from temp_algorithm_ import select_next_node

def read_tsp_data(file_path):
    """解析TSPLIB格式数据，返回坐标列表 [[6]]"""
    cities = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('NODE_COORD_SECTION'):
                break
        for line in f:
            if line.strip() == 'EOF':
                break
            parts = line.strip().split()
            cities.append((float(parts[1]), float(parts[2])))
    return cities

def calculate_distance_matrix(cities):
    """构建欧氏距离矩阵 [[4]]"""
    n = len(cities)
    return [
        [math.hypot(c1[0]-c2[0], c1[1]-c2[1]) for c2 in cities] 
        for c1 in cities
    ]

def is_feasible(path, num_cities):
    """验证TSP路径可行性 [[6]][[8]]"""
    if len(path) != num_cities + 1:
        return False
    if path[0] != path[-1]:
        return False
    visited = set(path[:-1])
    return len(visited) == num_cities and visited == set(range(num_cities))

def solve_tsp():
    # 加载数据
    cities = read_tsp_data(
        r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\berlin52.tsp"
    )
    distance_matrix = calculate_distance_matrix(cities)
    n = len(cities)
    
    # 修正1：确保起点和终点一致
    current_node = random.randint(0, n-1)
    destination_node = current_node  # 终点设为起点
    
    # 修正2：正确初始化未访问节点
    unvisited_nodes = [i for i in range(n) if i != current_node]
    random.shuffle(unvisited_nodes)
    
    path = [current_node]
    total_distance = 0
    
    while unvisited_nodes:
        next_node = select_next_node(
            current_node=current_node,
            destination_node=destination_node,
            unvisited_nodes=unvisited_nodes.copy(),
            distance_matrix=distance_matrix
        )
        total_distance += distance_matrix[current_node][next_node]
        path.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node
    
    # 闭合路径
    total_distance += distance_matrix[path[-1]][destination_node]
    path.append(destination_node)
    
    feasible = is_feasible(path, n)
    return path, total_distance, feasible

if __name__ == "__main__":
    path, distance, feasible = solve_tsp()
    
    # 创建结果字典
    result = {
        "status": feasible,
        "total_distance": distance,
        "path_length": len(path)
    }
    
    # 写入JSON文件
    try:
        with open(r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\tsp_result.json", 
                  'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("结果已成功写入JSON文件")
    except Exception as e:
        print(f"写入JSON文件失败: {str(e)}")
