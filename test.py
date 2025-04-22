# # from llama_cpp import Llama
# # import llama_cpp
# # import os
# # os.environ["GGML_CUDA_BLACKLIST"] = "0"   # 强制启用所有CUDA设备
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU编号
# # model_path = "D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

# # prompt="""     
# #          任务：生成TSP算法
# #          格式要求：
# #         1. 算法描述必须包裹在 <start> 和 <end> 标签之间，
# #         2. 代码必须定义为函数：def select_next_node(...):
# #         3. 禁止添加任何额外解释或注释

# #         输出示例：
# #         <start>优先选择距离近且密度高的节点<end>
# #         def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
# #             import numpy as np
# #             return next_node
# #         """

# # llm = Llama(model_path=model_path, n_gpu_layers=20, verbose=False)  # 注意开启verbose

# # response = llm(
# #     prompt=prompt,
# #     max_tokens=4096,
# #     temperature=0.1
# # )['choices'][0]['text']
# # print(response)
# import subprocess
# import os
# import logging
# data_path = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data"
# def load_evaluation_data():
#         """加载评估数据集和最优解"""
#         # 示例数据（需要根据实际数据调整）
#         instances = [
#             os.path.join(data_path, "eil101.tsp"),
#             os.path.join(data_path, "berlin52.tsp")
#         ]
#         optimals = [426, 7542]  # 对应实例的最优解
    
#         # 实际使用时可以从文件加载，例如：
#         # with open(os.path.join(self.data_path, "optimals.json")) as f:
#         #     optimals = json.load(f)
    
#         return instances, optimals
# eval_instances, optimal_solutions = load_evaluation_data()
# temp_dir = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"
# temp_file = f"temp_algorithm_.py"
# temp_path = os.path.join(temp_dir, temp_file)
# for instance, optimal in zip(eval_instances, optimal_solutions):
#                 # 验证实例文件存在
#                 if not os.path.exists(instance):
#                     logging.error(f"实例文件不存在：{instance}")
#                     continue
# process = subprocess.run(
#                         ["python", "tsp_evaluator.py", temp_path, instance],
#                         capture_output=True,
#                         text=True,
#                         timeout=60,
#                         check=True,
#                         env={**os.environ, "PYTHONPATH": r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"}
#                  
#    )
import re
import ast
import textwrap
import traceback
import json
import datetime
import logging

def parse_llm_response(response):
    try:
        # 清洗所有已知干扰标签和符号
        response = re.sub(r'</?(think|algorithm|code|explanation)[^>]*>', '', response, flags=re.IGNORECASE)
        response = re.sub(r':+start>', '<start>', response)  # 修复冒号开头的标签
        response = re.sub(r'```\w*|`', '', response)  # 去除所有代码标记
        response = re.sub(r'\n{3,}', '\n\n', response)  # 压缩多余空行
        response = re.sub(r'^\s*$\n', '', response, flags=re.MULTILINE)  # 删除空行

        # 支持多语言描述的灵活匹配（中英文混杂场景）
        desc_match = re.search(
            r'<start>\s*((?:[^<]|<(?!end>))*?)\s*<end>', 
            response, 
            flags=re.DOTALL|re.IGNORECASE
        )
        if not desc_match:
            raise ValueError("未找到有效的算法描述标签")
        description = desc_match.group(1).strip()
    
        # 增强代码块提取（兼容带/不带Markdown的情况）
        code_match = re.search(
            r'(?:```\w*?\n)?(def\s+select_next_node\s*\(.*?\):.*?)(?:```)?(?=\n\s*(?:def\s|class\s|#|$))',
            response,
            flags=re.DOTALL|re.IGNORECASE
        )
        if not code_match:
            raise ValueError("未找到有效的函数定义")
        code = code_match.group(1)

        # 参数名称强制统一化
        param_mapping = {
            r'\bcurrent[\s_]*node\b': 'current_node',
            r'\bdest[\s_]*node\b': 'destination_node',
            r'\bunvisited[\s_]*nodes?\b': 'unvisited_nodes',
            r'\bdist[\s_]*matrix\b': 'distance_matrix'
        }
        for pattern, replacement in param_mapping.items():
            code = re.sub(pattern, replacement, code, flags=re.IGNORECASE)

        # 自动修复常见语法错误
        code = _fix_code_syntax(code)
    
        # 确保必要的库导入
        if 'import numpy' not in code:
            code = 'import numpy as np\n' + code

        try:
            ast.parse(code)
        except SyntaxError as e:
            # 尝试二次修复
            repaired_code = _advanced_syntax_repair(code, str(e))
            ast.parse(repaired_code)
            code = repaired_code

        # 标准化缩进
        code = textwrap.dedent(code).strip()
        code = '\n'.join([line.rstrip() for line in code.split('\n')])
    
        return description, code

    except Exception as e:
        error_info = {
            "error": str(e),
            "response_snippet": response[:1000],
            "traceback": traceback.format_exc()
        }
        _log_parse_error(error_info)
        return None, None

def _fix_code_syntax(code):
    """多阶段语法修复"""
    # 修复中文标点
    code = code.translate(str.maketrans('：（）', ':()'))

    # 修复缺失冒号
    code = re.sub(r'(def\s+\w+\(.*?\))\s*$', r'\1:', code)

    # 修复未闭合括号
    open_brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for i, char in enumerate(code):
        if char in open_brackets:
            stack.append((char, i))
        elif char in open_brackets.values():
            if not stack or open_brackets[stack[-1][0]] != char:
                # 尝试自动修复
                code = code[:i] + open_brackets[stack[-1][0]] + code[i:]
            else:
                stack.pop()

    # 确保return语句存在
    if not re.search(r'return\s+\w+', code):
        code_lines = code.split('\n')
        for i, line in enumerate(reversed(code_lines)):
            if line.strip().startswith('def '):
                code_lines.insert(-i, '    return next_node')
                break
        code = '\n'.join(code_lines)

    return code

def _advanced_syntax_repair(code, error_msg):
    """基于错误信息的智能修复"""
    # 处理 'NoneType' has no attribute 'id' 类错误
    if "'NoneType'" in error_msg:
        code = re.sub(r'(\w+)\s*=\s*None\s*\n', r'\1 = 0\n', code)

    # 处理未定义变量
    undefined_var = re.search(r"name '(\w+)' is not defined", error_msg)
    if undefined_var:
        var_name = undefined_var.group(1)
        code = f"{var_name} = None\n" + code

    # 处理缩进错误
    if "unexpected indent" in error_msg:
        code = textwrap.dedent(code)

    return code

def _log_parse_error(error_info):
    """结构化错误日志记录"""
    log_entry = json.dumps({
        "timestamp": datetime.now().isoformat(),
        "type": "PARSE_ERROR",
        "details": error_info
    }, ensure_ascii=False)

    logging.error(log_entry)
    with open("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\log\\parse_llm_response_error.log", "a") as f:
        f.write(log_entry + "\n")

def _normalize_code(code):
    # 统一参数名称
    replacements = {
        r'\bcurrent_?node\b': 'current_node',
        r'\bdest_?node\b': 'destination_node',
        r'\bunvisited\b': 'unvisited_nodes',
        r'\bdist_?matrix\b': 'distance_matrix'
    }
    for pattern, repl in replacements.items():
        code = re.sub(pattern, repl, code, flags=re.IGNORECASE)

    # 确保必要库导入
    if 'import numpy' not in code:
        code = 'import numpy as np\n' + code
    return code
def load_algorithm(filepath):
    """从文件加载算法"""
    with open(filepath, "r",encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"<start>(.*?)<end>\n(.*)", content, re.DOTALL)
    print(match,'\n','\n')
    if match:
        logging.info("The algorithm has loaded")
        return {
            "description": match.group(1).strip(),
            "code": match.group(2).strip()
        }
    return None
a=load_algorithm(filepath="D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\algorithm1.txt")
print(a["description"],'\n')
print(a["code"],'\n')