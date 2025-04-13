import sys
import json
import math
import importlib.util
from typing import List

def calculate_distance(path: List[int], distance_matrix: List[List[float]]) -> float:
    """计算路径总距离（包含返回起点的距离）"""
    total = 0.0
    for i in range(len(path)-1):
        total += distance_matrix[path[i]][path[i+1]]
    # 添加回到起点的距离
    total += distance_matrix[path[-1]][path[0]]
    return total

def load_tsp_instance(file_path: str) -> List[List[float]]:
    """加载TSP实例文件并生成距离矩阵"""
    nodes = []
    with open(file_path, 'r') as f:
        read_coords = False
        for line in f:
            line = line.strip()
            if line.startswith('NODE_COORD_SECTION'):
                read_coords = True
                continue
            if line == 'EOF':
                break
            if read_coords and line:
                parts = line.split()
                nodes.append((float(parts[1]), float(parts[2])))
    
    # 计算距离矩阵
    dimension = len(nodes)
    distance_matrix = [[0.0]*dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i+1, dimension):
            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            dist = math.sqrt(dx**2 + dy**2)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

def validate_solution(path: List[int], dimension: int):
    """验证解的合法性"""
    if len(path) != dimension:
        raise ValueError(f"路径不完整，应包含{dimension}个节点，实际{len(path)}个")
    
    if sorted(path) != list(range(dimension)):
        raise ValueError("路径包含重复或非法节点")
    
def solve_tsp(distance_matrix: List[List[float]]) -> List[int]:
    """基于select_next_node的TSP求解主逻辑"""
    dimension = len(distance_matrix)
    current_node = 0
    destination_node = 0
    unvisited = set(range(1, dimension))
    path = [current_node]
    
    while unvisited:
        try:
            next_node = select_next_node(
                current_node=current_node,
                destination_node=destination_node,
                unvisited_nodes=list(unvisited),
                distance_matrix=distance_matrix
            )
            
            if next_node not in unvisited:
                raise ValueError(f"非法节点选择：{next_node}，剩余可选节点：{unvisited}")
            
            path.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        except Exception as e:
            raise RuntimeError(f"节点选择失败：{str(e)}")
    
    # 最后添加返回起点的步骤（可选）
    path.append(destination_node)
    return path

# 修改主程序参数处理
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({
            "status": "error",
            "message": "参数错误，需要2个参数：算法文件 实例文件"
        }))
        sys.exit(1)

    algorithm_file = sys.argv[1]
    instance_file = sys.argv[2]

    # 动态加载算法模块
    try:
        spec = importlib.util.spec_from_file_location("dynamic_algorithm", algorithm_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        select_next_node = module.select_next_node
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"算法加载失败：{str(e)}"
        }))
        sys.exit(1)