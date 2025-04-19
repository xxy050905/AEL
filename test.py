# from llama_cpp import Llama
# import llama_cpp
# import os
# os.environ["GGML_CUDA_BLACKLIST"] = "0"   # 强制启用所有CUDA设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU编号
# model_path = "D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

# prompt="""     
#          任务：生成TSP算法
#          格式要求：
#         1. 算法描述必须包裹在 <start> 和 <end> 标签之间，
#         2. 代码必须定义为函数：def select_next_node(...):
#         3. 禁止添加任何额外解释或注释
    
#         输出示例：
#         <start>优先选择距离近且密度高的节点<end>
#         def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
#             import numpy as np
#             return next_node
#         """

# llm = Llama(model_path=model_path, n_gpu_layers=20, verbose=False)  # 注意开启verbose

# response = llm(
#     prompt=prompt,
#     max_tokens=4096,
#     temperature=0.1
# )['choices'][0]['text']
# print(response)
import subprocess
import os
import logging
data_path = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data"
def load_evaluation_data():
        """加载评估数据集和最优解"""
        # 示例数据（需要根据实际数据调整）
        instances = [
            os.path.join(data_path, "eil101.tsp"),
            os.path.join(data_path, "berlin52.tsp")
        ]
        optimals = [426, 7542]  # 对应实例的最优解
        
        # 实际使用时可以从文件加载，例如：
        # with open(os.path.join(self.data_path, "optimals.json")) as f:
        #     optimals = json.load(f)
        
        return instances, optimals
eval_instances, optimal_solutions = load_evaluation_data()
temp_dir = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"
temp_file = f"temp_algorithm_.py"
temp_path = os.path.join(temp_dir, temp_file)
for instance, optimal in zip(eval_instances, optimal_solutions):
                # 验证实例文件存在
                if not os.path.exists(instance):
                    logging.error(f"实例文件不存在：{instance}")
                    continue
process = subprocess.run(
                        ["python", "tsp_evaluator.py", temp_path, instance],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=True,
                        env={**os.environ, "PYTHONPATH": r"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL"}
                    )