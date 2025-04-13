from llama_cpp import Llama
import llama_cpp
import os
os.environ["GGML_CUDA_BLACKLIST"] = "0"   # 强制启用所有CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU编号
model_path = "D:\\Deepseek\\llama.cpp\\models\\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

prompt="""     
        # 任务：生成TSP算法
        ## 格式要求：
        1. 算法描述必须包裹在 <start> 和 <end> 标签之间，例如：
        <start>这是一个结合距离和密度的创新算法<end>
        2. 代码必须定义为函数：def select_next_node(...):
        3. 禁止添加任何额外解释或注释
    
        ## 示例：
        <start>优先选择距离近且密度高的节点<end>
        def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
            import numpy as np
            # 计算密度分数...
            return next_node
        """
# llm=Llama(  
#     model_path=model_path,
#     n_ctx=131072,
#     n_threads=4,          # 旧版本建议减少CPU线程数
#     n_gpu_layers=75,      # 必须大于0才会启用GPU
#     n_batch=2048,         # 显存不足时可适当减小
#     use_mmap=True
#     verbose=False,
#     main_gpu=1)
# # 通过创建模型实例时观察日志输出
llm = Llama(model_path=model_path, n_gpu_layers=20, verbose=True)  # 注意开启verbose
# 如果看到类似"llm_load_tensors: using CUDA for GPU acceleration"的日志说明GPU已启用
# print(Llama(model_path=model_path,verbose=False).backend)  # 应该显示CUDA后端
# print(llama_cpp.llama_cublas_get())  # 返回1表示CUDA支持已编译
response = llm(
    prompt=prompt,
    max_tokens=4096,
    temperature=0.1
)['choices'][0]['text']
print(response)