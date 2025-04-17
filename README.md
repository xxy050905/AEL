原理概述:
基于Algorithm Evolution Using Large Language Model论文中提出的AEL框架,尝试设计关于TSP问题的新算法,
TSP问题的核心在于如何寻找下一节点,故本项目将使用LLM设计select next node函数并存于temp_algorithm.py文件中
并且在tsp_evaluation.py中导入并测试生成的函数是否可行

现阶段问题:
1.main.py可以正常运行,但代码生成的成功率很低,只有过一次成功写入的案例(但不符合提示词中的要求,已删除).根据日志,
LLM生成的代码似乎没有问题,在进化过程中还是发生错误,在提取LLM生成的文本中已经很少发生错误,目前猜测原因是在评价生
成的算法时新生成的select next node与原本temp algorithm中的select next node发生冲突.

解决方案:
使用临时文件,每次使用完temp_algorithm后直接删除,避免前后冲突(效果未知)

2.无法使用GPU运行(已在test.py中尝试),目前没想到方法解决,疑似llama_cpp版本的问题

3.使用.txt文件存储数据是否效率不高?可以换成.json文件存储数据,但如果更换,main.py要大量修改,现阶段暂时不考虑

4.在LLM生成的每个算法之间的相似度很高,暂时没有解决方法
