from llama_cpp import Llama
import llama_cpp
import os
os.environ["GGML_CUDA_BLACKLIST"] = "0"   # 强制启用所有CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU编号
model_path = "D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

prompt="""     
         任务：生成TSP算法
         格式要求：
        1. 算法描述必须包裹在 <start> 和 <end> 标签之间，
        2. 代码必须定义为函数：def select_next_node(...):
        3. 禁止添加任何额外解释或注释
    
        输出示例：
        <start>优先选择距离近且密度高的节点<end>
        def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
            import numpy as np
            return next_node
        """

llm = Llama(model_path=model_path, n_gpu_layers=20, verbose=False)  # 注意开启verbose

response = llm(
    prompt=prompt,
    max_tokens=4096,
    temperature=0.1
)['choices'][0]['text']
print(response)
