import sys
import math
import json
sys.path.append(r"D:\Paper\Algorithm Evolution Using Large Language Model\code\AEL")
# from temp_algorithm_ import select_next_node

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
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    total_nodes = len(distance_matrix)
    visited_ratio = 1 - len(unvisited_nodes)/total_nodes 
    if visited_ratio < 0.7:
        best_node = None
        max_score = -float('inf')
        for node in unvisited_nodes:
            valid_neighbors = sum(1 for n in unvisited_nodes 
                                 if n != node and distance_matrix[node][n] < distance_matrix[current_node][node])
            current_dist = distance_matrix[current_node][node]
            score = valid_neighbors * 100 + (1 / current_dist if current_dist != 0 else float('inf'))
            if score > max_score or (score == max_score and 
                                    distance_matrix[node][destination_node] < distance_matrix[best_node][destination_node]):
                max_score = score
                best_node = node
        return best_node
    else:
        return min(unvisited_nodes, 
                   key=lambda x: (distance_matrix[current_node][x] + 
                                  0.3 * distance_matrix[x][destination_node]))
def solve_tsp():
    # 加载数据
    cities = read_tsp_data(
        r"D:\Paper\Algorithm Evolution Using Large Language Model\code\AEL\data\berlin52.tsp"
    )
    distance_matrix = calculate_distance_matrix(cities)
    n = len(cities)
    
    # 初始化参数
    current_node = 0
    destination_node = 0
    unvisited_nodes = list(range(1, n))
    path = [current_node]
    total_distance = 0
    
    # 构建路径
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
    
    # 写入JSON文件（带异常处理）
    try:
        with open(r"D:\Paper\Algorithm Evolution Using Large Language Model\code\AEL\data\tsp_result.json", 
                  'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("结果已成功写入JSON文件")
    except Exception as e:
        print(f"写入JSON文件失败: {str(e)}")
