import numpy as np
from llama_cpp import Llama
import subprocess
import json
import re
import os
import glob
import os
import logging
import ast
import textwrap
from datetime import datetime
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)

class AEL_TSP:
    def __init__(self, model_path, data_path):
        print("Initializing Llama model...")
        # 初始化LLM
        self.llm=Llama(  
            model_path=model_path,
            n_ctx=131072,
            n_threads=16,           
            n_gpu_layers=75,       
            verbose=False,
            main_gpu=1)
        print("Model initialized successfully.")
        # AEL参数
        self.pop_size = 4
        self.generations = 10
        self.crossover_prob = 1.0
        self.mutation_prob = 0.2
        self.parents_num = 2
        
         # 文件路径设置
        self.data_path = data_path
        self.next_algorithm_num = self.get_next_algorithm_number()
        #获取评估数据
        self.eval_instances, self.optimal_solutions = self.load_evaluation_data()
    def create_initial_prompt(self):
        return """
        # 任务：生成TSP算法
        ## 格式要求：
        1. 算法描述必须包裹在 <start> 和 <end> 标签之间，例如：
        <start>这是一个结合距离和密度的创新算法<end>
        2. 代码必须定义为函数：def select_next_node(...):
        3. 禁止添加任何额外解释或注释
    
        output example：
        <start>优先选择距离近且密度高的节点<end>
        def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
            import numpy as np
            return next_node
        """

    def get_next_algorithm_number(self):
        """获取下一个算法文件的编号"""
        existing_files = glob.glob(os.path.join(self.data_path, "algorithm*.txt"))
        if not existing_files:
            return 1
        numbers = [int(os.path.basename(f).split("algorithm")[1].split(".")[0]) for f in existing_files]
        return max(numbers) + 1
    
    def save_algorithm(self, description, code):
        """保存算法到文件"""
        filename = f"algorithm{self.next_algorithm_num:2d}.txt"
        filepath = os.path.join(self.data_path, filename)
        with open(filepath, "w",encoding='utf-8') as f:
            f.write(f"<start>{description}<end>\n{code}")
        self.next_algorithm_num += 1
        return filepath
    
    def load_algorithm(self, filepath):
        """从文件加载算法"""
        with open(filepath, "r",encoding='utf-8') as f:
            content = f.read()
        match = re.search(r"<start>(.*?)<end>\n(.*)", content, re.DOTALL)
        if match:
            logging.info("The algorithm has loaded")
            return {
                "description": match.group(1).strip(),
                "code": match.group(2).strip()
            }
        return None

    def create_crossover_prompt(self, parents):
        prompt = """
        Task: Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. The task can be solved step-by-step by starting from the current node and iteratively choosing the next node.
        I have {num_parents} algorithms with their code to select the next node in each step.
        {parent_algorithms}
        Please help me create a new algorithm that is motivated by the given algorithms. Provide a brief description of the new algorithm and its corresponding code. The description must start with '<start>' and end with '<end>'. The code function must be called 'select_next_node' that takes inputs 'current_node', 'destination_node', 'unvisited_nodes', and 'distance_matrix', and outputs the 'next_node', where 'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node IDs.
        Be creative and do not give any explanation.
        do not give any explanation.
        output example：
        <start>优先选择距离近且密度高的节点<end>
        def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
            import numpy as np
            return next_node
        """
    
        # 填充父代算法信息
        parent_algorithms = ""
        for i, parent in enumerate(parents):
            parent_algorithms += f"\nAlgorithm {i+1}:\n"
            parent_algorithms += f"<Algorithm description>: {parent['description']}\n"
            parent_algorithms += f"<Code>: {parent['code']}\n\n"
    
        return prompt.format(
            num_parents=len(parents),
            parent_algorithms=parent_algorithms
        )
    def create_mutation_prompt(self, parent):
        """创建变异提示词"""
        return f"""
        Task: Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. The task can be solved step-by-step by starting from the current node and iteratively choosing the next node.

        I have an algorithm with its code to select the next node in each step.

        Algorithm description: {parent['description']}
        Code: {parent['code']}

        Please assist me in creating a MODIFIED VERSION of this algorithm. The new algorithm should:
        1. Maintain the core idea of the original algorithm
        2. Introduce at least one innovative modification (e.g. add new criteria, adjust parameters, or combine with other strategies)
        3. Keep the same input/output interface

        Provide a brief description of the modified algorithm and its corresponding code. The description must start with '<start>' and end with '<end>'. The code function must be called 'select_next_node' that takes inputs 'current_node', 'destination_node', 'unvisited_nodes', and 'distance_matrix', and outputs the 'next_node'.

        Focus on making meaningful improvements, not just code formatting changes.
        output example：
        <start>优先选择距离近且密度高的节点<end>
        def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
            import numpy as np
            return next_node
        """
    def parse_llm_response(self, response):
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
            code = self._fix_code_syntax(code)
        
            # 确保必要的库导入
            if 'import numpy' not in code:
                code = 'import numpy as np\n' + code

            try:
                ast.parse(code)
            except SyntaxError as e:
                # 尝试二次修复
                repaired_code = self._advanced_syntax_repair(code, str(e))
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
            self._log_parse_error(error_info)
            return None, None

    def _fix_code_syntax(self, code):
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

    def _advanced_syntax_repair(self, code, error_msg):
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

    def _log_parse_error(self, error_info):
        """结构化错误日志记录"""
        log_entry = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "type": "PARSE_ERROR",
            "details": error_info
        }, ensure_ascii=False)
    
        logging.error(log_entry)
        with open("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\log\\parse_llm_response_error.log", "a") as f:
            f.write(log_entry + "\n")

    def _normalize_code(self, code):
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

        
    def load_evaluation_data(self):
        """加载评估数据集和最优解"""
        # 示例数据（需要根据实际数据调整）
        instances = [
            os.path.join(self.data_path, "eil101.tsp"),
            os.path.join(self.data_path, "berlin52.tsp")
        ]
        optimals = [426, 7542]  # 对应实例的最优解
        
        # 实际使用时可以从文件加载，例如：
        # with open(os.path.join(self.data_path, "optimals.json")) as f:
        #     optimals = json.load(f)
        
        return instances, optimals
    
    def initialize_population(self):
        population = []
        files = glob.glob(os.path.join(self.data_path, "algorithm*.txt"))
        # 优先加载已有算法
        for f in files[:5]:
            algorithm = self.load_algorithm(f)
            if algorithm:
                algorithm["fitness"] = self.evaluate_algorithm(algorithm["code"])
                population.append(algorithm)
        logging.info("The existed algorithm has loaded successfully")
        
    # 如果种群不足，通过LLM生成补充
        while len(population) < self.pop_size:
            response = self.llm(prompt=self.create_initial_prompt())['choices'][0]['text']
            desc, code = self.parse_llm_response(response)
            if desc and code:
                fitness = self.evaluate_algorithm(code)
                population.append({
                    "description": desc,
                    "code": code,
                    "fitness": fitness
                })
                self.save_algorithm(desc, code)  # 保存新生成的算法
            if len(population) >= self.pop_size:
                logging.info("The supplementary algorithm has been generated successfully")
                break
        return population[:self.pop_size]

    def evaluate_algorithm(self, code):
        # 保存算法到临时文件（使用绝对路径）
        temp_dir = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"
        temp_file = f"temp_algorithm_.py"
        temp_path = os.path.join(temp_dir, temp_file)
    
        try:
            # 确保目录存在
            os.makedirs(temp_dir, exist_ok=True)
        
            # 写入代码文件
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(code + "\n\n")
                logging.info(f"成功写入临时文件：{temp_path}")
        
            for instance, optimal in zip(self.eval_instances, self.optimal_solutions):
                # 验证实例文件存在
                if not os.path.exists(instance):
                    logging.error(f"实例文件不存在：{instance}")
                    continue
                
                # 运行评估（传递正确的参数顺序）
                try:
                    process = subprocess.run(
                        ["python", "tsp_evaluator.py", temp_path, instance],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=True,
                        env={**os.environ, "PYTHONPATH": r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"}
                    )

                    # 解析结果
                    file_path = r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\tsp_result.json"

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)  # 正确使用文件对象
                            if result["status"]:
                                return result["status"]
                            else:
                                logging.info("The algorithm can not run")
                                return result["status"]
                        
                    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                        error_info = {
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "instance": instance,
                        "temp_file": temp_path,
                        "stdout": process.stdout if 'process' in locals() else '',
                        "stderr": process.stderr if 'process' in locals() else ''
                        }
                        logging.error(json.dumps(error_info, indent=2))
                except:
                    logging.info("evaluate algorithm failed")
        finally:
            # 确保清理临时文件
            try:
                os.remove(temp_path)
                os.remove("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\tsp_result.json")
                logging.info(f"已清理临时文件：{temp_path}")
            except Exception as e:
                logging.warning(f"临时文件清理失败：{str(e)}")

    def evolve(self):
        population = self.initialize_population()
        for gen in range(self.generations):
            new_population = []
            
            for _ in range(self.pop_size):
                # 选择
                parents = np.random.choice(population, self.parents_num, replace=False)
                
                # 交叉
                if np.random.rand() < self.crossover_prob:
                    logging.info("The algorithm is crossovering")
                    prompt = self.create_crossover_prompt(parents)
                    response = self.llm(
                        prompt=prompt,
                        max_tokens=4096,
                        temperature=0.1
                    )['choices'][0]['text']
                    logging.info(response)
                    desc, code = self.parse_llm_response(response)
                    if desc and code:
                        # 保存新算法
                        fitness = self.evaluate_algorithm(code)
                        if fitness:
                            self.save_algorithm(desc, code)
                            new_population.append({
                                "description": desc,
                                "code": code,
                                "fitness": fitness
                            })
                            logging.info("Algorithm has crossovered successfully")
                        else:
                            logging.info("The new algorithm can not run")
                    else:
                        logging.info("The algorithm crossovered failed")
                        error_msg = f"crossover failed\n{response}\n{'='*50}"
                        with open("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\log\\evolve_failed.log", "a") as f:
                            f.write(error_msg + "\n")
                
                # 变异（示例）
                for ind in list(new_population):  # 遍历新生成的个体
                    if np.random.rand() < self.mutation_prob:
                        try:
                            logging.info("The algorithm is trying mutate")
                            # 生成变异提示
                            prompt = self.create_mutation_prompt(ind)
                            
                            # 调用LLM生成变异算法
                            response = self.llm(
                                prompt=prompt,
                                max_tokens=4096,
                                temperature=0.1
                            )['choices'][0]['text']
                            
                            # 解析响应
                            desc, code = self.parse_llm_response(response)
                            if desc and code:
                                # 保存新算法
                                self.save_algorithm(desc, code)
                                
                                # 评估并加入种群
                                fitness = self.evaluate_algorithm(code)
                                new_population.append({
                                    "description": desc,
                                    "code": code,
                                    "fitness": fitness
                                })
                                logging.info("The algorithm has mutated successfully")
                        except Exception as e:
                            print(f"Mutation error: {str(e)}")
            # 种群管理
            population = sorted(population + new_population, key=lambda x: x['fitness'])[:self.pop_size]
        logging.info("evolve func is done")

# 使用示例
if __name__ == "__main__":
    model_path = "D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    data_path = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data"
    
    ael = AEL_TSP(model_path, data_path)
    best_algorithm = ael.evolve()