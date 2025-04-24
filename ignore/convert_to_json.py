import json
import os

algorithms = []

for i in range(1, 27):
    filename = f"D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\algorithm{i}.txt"
    if not os.path.exists(filename):
        continue
        
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        desc_start = content.find('<start>') + len('<start>')
        desc_end = content.find('<end>')
        description = content[desc_start:desc_end].strip()
        code = content[desc_end+len('<end>'):].strip()

        algorithms.append({
            "id": i,
            "description": description,
            "code": code
        })

with open('D:\\Paper\\Algorithm Evolution Using Large Language Model\\code\\AEL\\data\\algorithms.json', 'w', encoding='utf-8') as f:
    json.dump(algorithms, f, ensure_ascii=False, indent=2)