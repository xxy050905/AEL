原理概述:
基于Algorithm Evolution Using Large Language Model论文中提出的AEL框架,尝试设计关于TSP问题的新算法,
TSP问题的核心在于如何寻找下一节点,故本项目将使用LLM设计select next node函数并存于temp_algorithm.py文件中
并且在tsp_evaluation.py中导入并测试生成的函数是否可行,如果想要将设计的算法不局限于旅行商问题,可以通过LLM来生成解决问题的
代码模版,再让LLM生成并迭代出能够解决这个问题的核心函数

现阶段问题:
1.main.py可以正常运行,但代码生成的成功率很低,只有过两次成功写入的案例.根据日志,LLM生成的代码似乎没有问题,在进化过程的交叉
阶段总是失败,目前猜测是parse_llm_response函数或algorithm_evaluate函数出现问题导致交叉失败

2.无法使用GPU运行(已在test.py中尝试),目前没想到方法解决,疑似llama_cpp版本的问题
    
未来优化方案

1.使用.txt文件存储数据是否效率不高?可以换成.json文件存储数据.


