import re
import textwrap

def parse_llm_response(response):
    """解析LLM响应，提取算法描述和代码，强制参数标准化"""
    try:
        # ============ 预处理阶段 ============

        # 1) 用非贪婪方式删除 ```...``` 整个代码块
        response = re.sub(r'```[\s\S]*?```', '', response)
        response = response.replace('\u200b', '')

        # ============ 描述提取 ============
        desc_match = re.search(r'<start>(.*?)<end>', response, flags=re.DOTALL|re.IGNORECASE)
        if not desc_match:
            raise ValueError("未找到描述标签")
        description = desc_match.group(1).strip()

        # ============ 代码提取 ============
        code_pattern = re.compile(
            r'def\s+select_next_node\s*\(\s*([^)]*)\)\s*:\s*'   # 函数签名
            r'((?:\n[ \t].*?)*)'                                # 缩进的函数体
            r'(?=\n[^ \t]|\Z)',                                # 停在下一个顶格行或末尾
            flags=re.DOTALL|re.IGNORECASE
        )
        code_match = code_pattern.search(response)
        if not code_match:
            print("调试信息：响应内容截取\n", response[:500])
            raise ValueError("函数定义匹配失败")

        # ============ 参数标准化 ============
        raw_params = [p.strip() for p in code_match.group(1).split(',') if p.strip()]
        param_map = ['current_node','destination_node','unvisited_nodes','distance_matrix']
        if len(raw_params) != 4:
            raise ValueError(f"参数数量错误（需要4个，实际{len(raw_params)}个）")

        # ============ 代码重构 ============
        func_body = code_match.group(2)
        # 确保有 numpy 导入
        if 'import numpy' not in func_body:
            func_body = '\n    import numpy as np' + func_body

        # 拼回标准化函数
        standardized_code = (
            f"def select_next_node({', '.join(param_map)}):\n"
            + func_body
        )

        return description, standardized_code

    except Exception as e:
        print(f"解析错误: {str(e)}")
        return None, None


# 测试用例
test_response = """
<start>综合算法1和算法2的动态加权选择标准<end>
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    import numpy as np
    # 算法1：选择距离最近的节点
    min_dist = float('inf')
    nearest = unvisited_nodes[0]
    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_dist:
            min_dist = distance_matrix[current_node][node]
            nearest = node
    # 算法2：选择密度最高的节点
    density_scores = []
    for node in unvisited_nodes:
        count = 0
        for neighbor in unvisited_nodes:
            if distance_matrix[node][neighbor] < 30:
                count +=1
        density_scores.append(count)
    # 综合权重，距离60%，密度40%
    scores = [min_dist * 0.6 + density_scores[i] * 0.4 for i in range(len(unvisited_nodes))]
    return unvisited_nodes[np.argmin(scores)]
</start>
"""

desc, code = parse_llm_response(test_response)
print(f"描述: {desc}\n") 
print(f"代码:\n{code}")