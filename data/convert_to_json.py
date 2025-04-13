import json
import os
import re
import glob

def txt_to_json(data_path, output_file="algorithms.json"):
    """
    将algorithm*.txt文件转换为JSON格式
    参数说明：
    - data_path: 原始txt文件存储路径
    - output_file: 输出JSON文件名
    """
    algorithms = []
    
    # 匹配文件模式
    file_pattern = os.path.join(data_path, "algorithm*.txt")
    
    # 解析每个txt文件
    for txt_file in glob.glob(file_pattern):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 使用改进的正则表达式解析
            match = re.search(
                r"<start>(.*?)<end>\s*(def select_next_node.*?)(?=\n<|\Z)",
                content, 
                re.DOTALL | re.IGNORECASE
            )
            
            if match:
                # 提取编号
                file_id = int(re.search(r"algorithm(\d+)", txt_file).group(1))
                
                algorithms.append({
                    "id": file_id,
                    "description": match.group(1).strip(),
                    "code": match.group(2).strip()
                })
                
        except Exception as e:
            print(f"转换失败 {os.path.basename(txt_file)}: {str(e)}")
    
    # 按原始编号排序
    algorithms.sort(key=lambda x: x["id"])
    
    # 保存为JSON
    output_path = os.path.join(data_path, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(algorithms, f, indent=2, ensure_ascii=False)
    
    print(f"成功转换 {len(algorithms)} 个算法到 {output_path}")

if __name__ == "__main__":
    data_path = "D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\TSP\\data"
    txt_to_json(data_path)