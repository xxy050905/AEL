import numpy as np
from llama_cpp import Llama
import subprocess
import json
import jsonschema
import re
import os
import glob
import os
import logging
import ast
import textwrap
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import time
from collections import defaultdict
import hashlib
from tempfile import NamedTemporaryFile


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)

class AEL_TSP:
    def __init__(self, model_path, data_path):
        print("Initializing Llama model...")
        # 初始化LLM
        self.llm=Llama(  
            model_path=model_path,
            n_ctx=131072,
            n_thread=16,        
            n_gpu_layers=-1,       
            verbose=False,
            main_gpu=1)
        print("Model initialized successfully.")
        # AEL参数
        self.pop_size = 30
        self.generations = 5
        self.crossover_num = 5
        self.crossover_prob = 1.0
        self.mutation_prob = 0.2
        self.parents_num = 10
        
         # 文件路径设置
        self.data_path = data_path
        self.next_algorithm_num = self.get_next_algorithm_number()
        #获取评估数据
        self.eval_instances, self.optimal_solutions = self.load_evaluation_data()
        #成功率统计
        self.success_stats = {
        'initialization': {'success': 0, 'fail': 0},
        'crossover': {'success': 0, 'fail': 0},
        'mutation': {'success': 0, 'fail': 0},
        'total_attempts': 0,
        'generation_stats': [],
        'error_types': {
            'parse_error': 0,
            'invalid_code': 0,
            'runtime_error': 0,
            'other_errors': 0
        }
    }
    def get_next_algorithm_number(self):
        """获取下一个算法ID（基于JSON文件）"""
        json_file = r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\algorithms.json"
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    algorithms = json.load(f)
                    return max(a['id'] for a in algorithms) + 1 if algorithms else 1
            return 1
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON解析错误: {str(e)}")
            return 1
    
    def save_algorithm(self, description, code):
        """保存算法到JSON文件"""
        json_file = r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\algorithms.json"
        new_algorithm = {
            "id": self.next_algorithm_num,
            "description": description,
            "code": code,
        }

        # 读取现有数据或初始化新文件
        try:
            existing_data = []
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            existing_data.append(new_algorithm)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            logging.error(f"保存失败: {str(e)}")
            traceback.print_exc()
        
        self.next_algorithm_num += 1
        return new_algorithm['id']
    
    def load_algorithm(self, algorithm_id):
        """加载指定ID的算法"""
        json_file = os.path.join(self.data_path, "algorithms.json")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                algorithms = json.load(f)
                for algo in algorithms:
                    if algo['id'] == algorithm_id:
                        return {
                            "id": algo['id'],
                            "description": algo['description'],
                            "code": algo['code']
                        }
            return None
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"加载算法失败: {str(e)}")
            return None

    def create_initial_prompt(self):
        return"""
            find the shortest route that visits each node once and returns to
            the starting node. The task can be solved step-by-step by starting
            from the current node and iteratively choosing the next node.
            You should create a totally new strategy for me (different from
            the heuristics in the literature) to select the next node in each step,
            using information including the current node, destination node,
            unvisited nodes, and distances between them.
            Provide a brief description of the new algorithm and its
            corresponding code. The description must start with ‘<start>’ and
            end with ‘<end>’. The code function must called
            'select_next_node' that takes inputs 'current_node',
            'destination_node', 'unvisited_nodes', and 'distance_matrix', and
            outputs the 'next_node', where 'current_node', 'destination_node',
            'next_node', and 'unvisited_nodes' are node id.
            Be creative and do not give additional explanation
            and Do not duplicate with the parent algorithm
            output template：
            <start>Description of new algorithm<end>
            def select_next_node((current_node, destination_node, unvisited_nodes, distance_matrix):
                return unvisited_nodes[np.argmin(scores)]
            """
    def create_crossover_prompt(self, parents):
        prompt = """
            Task：hortest route that visits each node once and returns to the
            starting node. The task can be solved step-by-step by starting from
            the current node and iteratively choosing the next node.
            I have {num_parents}algorithms with their code to select the next node in each
            step parent algorithms:
            {parent_algorithms}
            
           Please help me create a new algorithm that motivated by the given
            algorithms. Please provide a brief description of the new algorithm
            and its corresponding code. The description must start with ‘<start>’
            and end with ‘<end>’. The code function must called
            'select_next_node' that takes inputs 'current_node', 'destination_node',
            'unvisited_nodes', and 'distance_matrix', and outputs the 'next_node',
            where 'current_node', 'destination_node', 'next_node', and
            'unvisited_nodes' are node id.
            Be creative and do not give additional explanation.Do not duplicate with the parent algorithm

            correct example：

            output template：
            <start>Description of new algorithm<end>
            def select_next_node((current_node, destination_node, unvisited_nodes, distance_matrix):
                return unvisited_nodes[np.argmin(scores)]
        """
    
        # 填充父代算法信息
        parent_algorithms = ""
        for i, parent in enumerate(parents):
            parent_algorithms += f"\nAlgorithm {i+1}:\n"
            parent_algorithms += f"<Algorithm description>: {parent['description']}\n"
            parent_algorithms += f"<Code>: {parent['code']}\n\n"
        prompt += "\nExample of the current best algorithm：\n"
        return prompt.format(
            num_parents=len(parents),
            parent_algorithms=parent_algorithms
        )

    def create_mutation_prompt(self, parent):
        """创建变异提示词"""
        return f"""
        Task: Given a set of nodes with their coordinates, you need to find
        the shortest route that visits each node once and returns to the
        starting node. The task can be solved step-by-step by starting from
        the current node and iteratively choosing the next node.
        I have an algorithm with its code to select the next node in each step.
        Algorithm Description:
        Code: {parent['code']}

        Please assist me in creating a modified version of the algorithm
        provided. Please provide a brief description of the new algorithm and
        its corresponding code. The description must start with ‘<start>’ and
        end with ‘<end>’. The code function must called 'select_next_node'
        that takes inputs 'current_node', 'destination_node', 'unvisited_nodes',
        and 'distance_matrix', and outputs the 'next_node', where
        'current_node', 'destination_node', 'next_node', and 'unvisited_nodes'
        are node id.
        Be creative and do not give additional explanation.
        output example：
        <start>Dynamic weighting algorithm integrating distance and density<end>
            def select_next_node((current_node, destination_node, unvisited_nodes, distance_matrix):
                distances = distance_matrix[current_node]
                densities = [len([n for n in unvisited_nodes if distance_matrix[n][m] < 30]) for m in unvisited_nodes]
                scores = [0.6*distances[i] + 0.4*densities[i] for i in range(len(unvisited_nodes))]
                return unvisited_nodes[np.argmin(scores)]
        """
    def parse_llm_response(self,response):
        """
        解析LLM响应，提取<start>…<end>中的描述和select_next_node函数，
        并标准化参数、插入numpy导入。
        返回 (description, standardized_code) 或 (None, None)。
        """
        def extract_function_block(text, name):
            """
            按“行＋缩进深度”提取以 def <name> 开头的整个函数（含装饰器）。
            """
            lines = text.splitlines(keepends=True)
            out, i = [], 0
            while i < len(lines):
                line = lines[i]
                if line.lstrip().startswith(f"def {name}") or (
                line.lstrip().startswith("@") and
                i+1 < len(lines) and
                lines[i+1].lstrip().startswith(f"def {name}")
                ):
                    indent0 = len(line) - len(line.lstrip())
                    # 收集装饰器与签名行
                    while i < len(lines) and (lines[i].lstrip().startswith("@") or lines[i].lstrip().startswith("def")):
                        out.append(lines[i]); i += 1
                    # 收集函数体：空行或缩进 > indent0
                    while i < len(lines):
                        cur = lines[i]
                        ind = len(cur) - len(cur.lstrip())
                        if cur.strip() == "" or ind > indent0:
                            out.append(cur); i += 1
                        else:
                            break
                    break
                i += 1
            return "".join(out)

        try:
            # 1) 删除所有 ```…``` code-fence
            response = re.sub(r'```[\s\S]*?```', '', response)
            # 2) 去除 <code> 标签，防止干扰提取
            response = re.sub(r'</?code>', '', response, flags=re.IGNORECASE)

            # 3) 抽取描述
            m_desc = re.search(r'<start>(.*?)<end>', response, flags=re.DOTALL|re.IGNORECASE)
            if not m_desc:
                raise ValueError("未找到<start>…<end>描述")
            description = m_desc.group(1).strip()

            # 4) 提取函数块（行+缩进算法）
            raw_block = extract_function_block(response, "select_next_node")
            if not raw_block.strip():
                raise ValueError("函数定义匹配失败")

            # 5) 签名匹配
            m_sig = re.match(r'\s*def\s+select_next_node\s*\(+'      # 一或多个“(”
                            r'\s*([^)]*?)\s*'                       # 捕获最内层参数
                            r'\)+\s*:', raw_block)
            if not m_sig:
                raise ValueError("函数签名解析失败")
            raw_params = [p.strip() for p in m_sig.group(1).split(',') if p.strip()]
            if len(raw_params) != 4:
                raise ValueError(f"参数数目错误，需4个，实得{len(raw_params)}个")
            std_params = ['current_node','destination_node','unvisited_nodes','distance_matrix']

            # 6) 函数体与 numpy 导入
            body = raw_block[m_sig.end():]
            if 'import numpy' not in body:
                body = '\n    import numpy as np' + body

            # 7) 重组标准化函数
            standardized_code = (
                f"def select_next_node({', '.join(std_params)}):\n"
                + textwrap.indent(body.lstrip('\n'), '    ')
            )
            return description, standardized_code

        except Exception as e:
            # 写日志
            log = {"time": datetime.now().isoformat(), "error": str(e)}
            self.success_stats['error_types']['parse_error'] += 1
            with open("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\log\\parse_llm_response_error.log","a") as f: 
                f.write(json.dumps(log)+"\n")
            print(f"解析错误: {e}")
            return None, None


    def _log_parse_error(self, error_info):
        """结构化错误日志记录"""
        log_entry = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "type": "PARSE_ERROR",
            "details": error_info
        }, ensure_ascii=False)
    
        # logging.error(log_entry)
        with open("D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\log\\parse_llm_response_error.log", "a") as f:
            f.write(log_entry + "\n")
        
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
        """初始化种群"""
        json_file = os.path.join(self.data_path, "algorithms.json")
        population = []
        
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    algorithms = json.load(f)
                    # 按ID倒序获取最新算法
                    sorted_algorithms = sorted(algorithms, key=lambda x: x['id'], reverse=False)
                    for algo in sorted_algorithms[:self.pop_size]:
                        algo_data = {
                            "description": algo['description'],
                            "code": algo['code'],
                            "fitness": self.evaluate_algorithm(algo['code'])
                        }
                        population.append(algo_data)
            logging.info("种群初始化成功，加载算法数量: {}".format(len(population)))
        except Exception as e:
            logging.error("种群初始化失败: {}".format(str(e)))
        
        while len(population) < self.pop_size:
            response = self.llm(prompt=self.create_initial_prompt(),
                                temperature=0.2,
                                max_tokens=2048
                                )['choices'][0]['text']
            print(response)
            desc, code = self.parse_llm_response(response)
            if desc and code:
                fitness = self.evaluate_algorithm(code)
                if fitness:
                    population.append({
                        "description": desc,
                        "code": code,
                        "fitness": fitness
                    })
                    self.success_stats['initialization']['success'] += 1
                    self.save_algorithm(desc, code)  # 保存新生成的算法
                else:
                    self.success_stats['initialization']['fail'] += 1
            if len(population) >= self.pop_size:
                logging.info("The supplementary algorithm has been generated successfully")
                break
        return population
    def visualize_success_rate(self, save_path=None):
        """可视化成功率并保存为图片"""
        plt.figure(figsize=(15, 10))
        
        # 成功率趋势图
        plt.subplot(2, 2, 1)
        generations = [x['generation'] for x in self.success_stats['generation_stats']]
        success_rates = [x['success_rate'] for x in self.success_stats['generation_stats']]
        plt.plot(generations, success_rates, marker='o')
        plt.title('Success Rate Trend per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Success Rate (%)')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        plt.grid(True)

        # 操作类型分布
        plt.subplot(2 ,2, 2)
        labels = ['Initialization', 'Crossover', 'Mutation']
        success = [
            self.success_stats["initialization"]["success"],
            self.success_stats['crossover']['success'],
            self.success_stats['mutation']['success']
        ]
        fails = [
            self.success_stats["initialization"]["fail"],
            self.success_stats['crossover']['fail'],
            self.success_stats['mutation']['fail']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, success, width, label='Success')
        plt.bar(x + width/2, fails, width, label='Failure')
        plt.title('Success/Failure by Operation Type')
        plt.xticks(x, labels)
        plt.legend()
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

      # 错误类型分布
        plt.subplot(2, 2, 3)
        error_types = self.success_stats['error_types']
        labels = list(error_types.keys())
        sizes = list(error_types.values())
        
        # 过滤掉0值的错误类型
        filtered_labels = [label for label, size in zip(labels, sizes) if size > 0]
        filtered_sizes = [size for size in sizes if size > 0]
        
        if filtered_sizes:
            plt.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%')
            plt.title('Error Type Distribution')
        else:
            plt.text(0.5, 0.5, 'No Errors Recorded', 
                    ha='center', va='center')
            plt.title('Error Type Distribution')


        # 总体统计
        plt.subplot(2, 2, 4)
        total_success = sum([v['success'] for k,v in self.success_stats.items() if isinstance(v, dict) and 'success' in v])
        total_fail = sum([v['fail'] for k,v in self.success_stats.items() if isinstance(v, dict) and 'fail' in v])
        plt.bar(['Success', 'Failure'], [total_success, total_fail], color=['green', 'red'])
        plt.title(f'Overall Success Rate: {total_success/(total_success+total_fail)*100:.1f}%')
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'success_rate_analysis.png'), bbox_inches='tight')
            logging.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close()

    def evaluate_algorithm(self, code):
        # 保存算法到临时文件
        temp_dir = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"
        temp_file = f"temp_algorithm_.py"
        temp_path = os.path.join(temp_dir, temp_file)
        valid_result = False

        try:
            # 确保目录存在
            os.makedirs(temp_dir, exist_ok=True)
            
            # 写入代码文件
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write("import numpy as np\n")  # 确保numpy可用
                f.write(code + "\n\n")
                logging.info(f"成功写入临时文件：{temp_path}")

            # 验证所有评估实例
            valid_count = 0
            for instance, optimal in zip(self.eval_instances, self.optimal_solutions):
                # 验证实例文件存在
                if not os.path.exists(instance):
                    logging.error(f"实例文件不存在：{instance}")
                    continue
                    
                try:
                    # 运行评估
                    process = subprocess.run(
                        ["python", "tsp_evaluator.py", temp_path, instance],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=True,
                        env={**os.environ, "PYTHONPATH": r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"}
                    )
                    
                    # 解析结果文件
                    result_file = r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\tsp_result.json"
                    if os.path.exists(result_file):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            if result.get("status") == 1:  # 有效解
                                valid_count += 1
                            else:  # 无效解
                                self.success_stats['error_types']['invalid_code'] += 1
                    else:  # 结果文件未生成
                        self.success_stats['error_types']['runtime_error'] += 1
                        logging.error(f"结果文件未生成: {instance}")

                except subprocess.CalledProcessError as e:
                    # 子进程返回非零状态码
                    self.success_stats['error_types']['invalid_code'] += 1
                    error_info = {
                        "type": "INVALID_CODE",
                        "instance": os.path.basename(instance),
                        "exit_code": e.returncode,
                        "stderr": e.stderr[:200]  # 截取前200字符
                    }
                    logging.error(json.dumps(error_info))
                    
                except subprocess.TimeoutExpired:
                    self.success_stats['error_types']['runtime_error'] += 1
                    logging.error(f"评估超时: {instance}")

            # 计算成功率
            success_rate = valid_count / len(self.eval_instances) if self.eval_instances else 0
            return success_rate

        except json.JSONDecodeError as e:
            self.success_stats['error_types']['runtime_error'] += 1
            logging.error(f"JSON解析失败: {str(e)}")
            return 0
            
        except Exception as e:
            self.success_stats['error_types']['other_errors'] += 1
            error_info = {
                "type": "UNEXPECTED_ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            logging.error(json.dumps(error_info, indent=2))
            return 0
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logging.info(f"已清理临时文件：{temp_path}")
            except Exception as e:
                self.success_stats['error_types']['other_errors'] += 1
                logging.warning(f"临时文件清理失败：{str(e)}")
            
            # 最终有效性检查
            if not valid_result:
                self.success_stats['error_types']['invalid_code'] += 1

    # def Multiple_test(self, data_path):
    #     """多数据集测试并返回最优算法"""
    #     # 1. 加载待测试算法
    #     algorithms = []
    #     json_file = os.path.join(data_path, "algorithms.json")
    #     try:
    #         with open(json_file, "r", encoding="utf-8") as f:
    #             algorithms = json.load(f)
    #     except Exception as e:
    #         logging.error(f"加载算法失败: {str(e)}")
    #         return None

    #     # 2. 获取所有TSP文件
    #     tsp_files = []
    #     for root, dirs, files in os.walk(data_path):
    #         for file in files:
    #             if file.lower().endswith('.tsp'):
    #                 tsp_files.append(os.path.join(root, file))
    #     if not tsp_files:
    #         logging.error("未找到任何TSP数据集文件")
    #         return None

    #     # 3. 准备临时目录
    #     temp_dir = os.path.join(data_path, "temp_algorithms_")
    #     os.makedirs(temp_dir, exist_ok=True)
    #     results = defaultdict(lambda: defaultdict(list))

    #     # 4. 评估每个算法
    #     for alg in algorithms:
    #         alg_id = alg["id"]
    #         code = alg.get("code", "")
            
    #         # 生成唯一文件名
    #         file_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    #         temp_filename = f"algorithm_{alg_id}_{file_hash}.py"
    #         temp_path = os.path.join(temp_dir, temp_filename)
            
    #         try:
    #             # 写入算法文件
    #             with open(temp_path, "w", encoding="utf-8") as f:
    #                 f.write("import numpy as np\n")
    #                 f.write(code)
                
    #             # 评估每个数据集
    #             for tsp_file in tsp_files:
    #                 output_file = os.path.join(temp_dir, f"result_{alg_id}_{os.path.basename(tsp_file)}.json")
                    
    #                 try:
    #                     # 运行评估程序
    #                     process = subprocess.run(
    #                         ['python', 'tsp_evaluator.py', temp_path, tsp_file, '--output', output_file],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.STDOUT,
    #                         timeout=120,
    #                         check=True
    #                     )
                        
    #                     # 解析结果
    #                     if os.path.exists(output_file):
    #                         with open(output_file, 'r') as f:
    #                             result = json.load(f)
    #                             if result['status']:
    #                                 results[alg_id]['distances'].append(result['total_distance'])
    #                 except subprocess.TimeoutExpired:
    #                     logging.warning(f"评估超时: 算法 {alg_id} - {tsp_file}")
    #                 except Exception as e:
    #                     logging.error(f"评估失败: {str(e)}")
            
    #         finally:
    #             # 清理临时文件
    #             if os.path.exists(temp_path):
    #                 try:
    #                     os.remove(temp_path)
    #                 except:
    #                     pass

    #     # 5. 计算综合评分
    #     scores = {}
    #     for alg_id, data in results.items():
    #         if data['distances']:
    #             scores[alg_id] = np.mean(data['distances'])
        
    #     if not scores:
    #         logging.error("没有有效的评估结果")
    #         return None

    #     # 6. 选择最优算法
    #     best_alg_id = min(scores, key=scores.get)
    #     best_alg = next(alg for alg in algorithms if alg["id"] == best_alg_id)
        
    #     # 7. 保存结果
    #     best_info = {
    #         "id": best_alg_id,
    #         "description": best_alg.get("description", ""),
    #         "average_distance": scores[best_alg_id],
    #         "sample_result": {
    #             "status": True,
    #             "total_distance": scores[best_alg_id],
    #             "path_length": len(best_alg.get("code", "").splitlines())
    #         }                      
    #     }
        
    #     result_file = os.path.join(data_path, "tsp_result.json")
    #     with open(result_file, "w", encoding="utf-8") as f:
    #         json.dump(best_info['sample_result'], f, indent=2)
        
    #     return best_info


    def evolve(self):
        # 初始化种群
        population = self.initialize_population()
    
        # 代数循环统计
        for gen in range(self.generations):
            new_population = []
            logging.info(f"=== 正在进化第 {gen+1}/{self.generations} 代 ===")
        
            # 代数级统计初始化
            gen_success = 0
            gen_attempts = 0
        
            for _ in range(self.crossover_num):
                # 选择父代
                parents = np.random.choice(population, self.parents_num, replace=False)
            
                # --- 交叉操作 ---
                if np.random.rand() < self.crossover_prob:
                    logging.info("执行交叉操作...")
                    try:
                        # 生成提示并调用LLM
                        prompt = self.create_crossover_prompt(parents)
                        response = self.llm(
                            prompt=prompt,
                            max_tokens=2048,
                            temperature=0.1,
                            frequency_penalty=0.5
                        )['choices'][0]['text']

                        # 解析响应
                        desc, code = self.parse_llm_response(response)
                        #交叉操作统计
                        if desc and code:
                            # 评估适应度
                            fitness = self.evaluate_algorithm(code)
                            if fitness:
                                # 保存算法）
                                self.save_algorithm(desc, code)
                                new_population.append({
                                    "description": desc,
                                    "code": code,
                                    "fitness": fitness
                                })
                            
                                # 统计成功
                                self.success_stats['crossover']['success'] += 1
                                gen_success += 1
                                logging.info("交叉成功！生成新算法")
                            else:
                                # 统计无效算法
                                self.success_stats['crossover']['fail'] += 1
                                logging.warning("交叉生成无效算法")
                        else:
                            # 统计解析失败
                            self.success_stats['crossover']['fail'] += 1
                            logging.error("交叉响应解析失败")
                    
                        # 统计总尝试次数
                        gen_attempts += 1
                        self.success_stats['total_attempts'] += 1
                
                    except Exception as e:
                        # 异常处理统计
                        self.success_stats['crossover']['fail'] += 1
                        logging.error(f"交叉操作异常: {str(e)}")
                        traceback.print_exc()

            # --- 变异操作 ---

                for ind in list(new_population):
                    if np.random.rand() < self.mutation_prob:
                        logging.info("执行变异操作...")
                        try:
                            # 生成提示并调用LLM
                            prompt = self.create_mutation_prompt(ind)
                            response = self.llm(
                                prompt=prompt,
                                max_tokens=2048,
                                temperature=0.1,
                                frequency_penalty=0.8
                            )['choices'][0]['text']
                            # print(response)
                            # 解析响应
                            desc, code = self.parse_llm_response(response)
                            # print("变异后结果:\n")
                            # print(desc,'\n')
                            # print(code,'\n')
                            # 变异操作统计
                            if desc and code:
                                # 评估适应度
                                fitness = self.evaluate_algorithm(code)
                                if fitness:
                                    # 保存算法
                                    self.save_algorithm(desc, code)
                                    new_population.append({
                                        "description": desc,
                                        "code": code,
                                        "fitness": fitness
                                    })
                                
                                    # 统计成功
                                    self.success_stats['mutation']['success'] += 1
                                    gen_success += 1
                                    logging.info("变异成功！生成新算法")
                                else:
                                    # 统计无效算法
                                    self.success_stats['mutation']['fail'] += 1
                                    logging.warning("变异生成无效算法")
                            else:
                                # 统计解析失败
                                self.success_stats['mutation']['fail'] += 1
                                logging.error("变异响应解析失败")
                        
                            # 统计总尝试次数
                            gen_attempts += 1
                            self.success_stats['total_attempts'] += 1
                    
                        except Exception as e:
                            # 异常处理统计
                            self.success_stats['mutation']['fail'] += 1
                            logging.error(f"变异操作异常: {str(e)}")
                            traceback.print_exc()

                population = sorted(population + new_population, key=lambda x: x['fitness'])[:self.pop_size]
        
            # 记录每代统计
            if gen_attempts > 0:
                success_rate = (gen_success / gen_attempts) * 100
            else:
                success_rate = 0.0
            
            self.success_stats['generation_stats'].append({
                'generation': gen+1,
                'attempts': gen_attempts,
                'success': gen_success,
                'success_rate': success_rate
            })
            logging.info(f"第 {gen+1} 代统计 - 尝试次数: {gen_attempts} | 成功率: {success_rate:.1f}%")
    
        # 进化完成后自动生成可视化
        logging.info("正在生成成功率分析图表...")    
        self.visualize_success_rate(save_path="D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\picture")
        logging.info("进化完成！")
        


# 使用示例
if __name__ == "__main__":
    model_path ="D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    data_path = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data"
    
    ael = AEL_TSP(model_path, data_path)
    ael.evolve()