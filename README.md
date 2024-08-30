# Medical-asistant
<1 功能简介：基于Yuan2-2B-Mars-hf大模型，结合RAG技术导入医学专业知识的健康小助手，集病症诊疗与就诊建议于一体，依据症状描述给出诊断建议，推荐相应的科室和对应检查项目，帮助患者明确就医方向，辅助专业诊断。并给出健康生活小贴士，帮助健康管理。
	 你是否遇到过身体不适，却不知严重与否、是否该就医、该挂哪科，网上查还易被误导的情况？别担心，智能诊疗助手来帮你！作为专属健康顾问，若有不适，只需简单描述症状，它能解析并提供可能的健康问题、检查项目、治疗方案和生活建议，让你心中有数。若有检查报告，它能提取内容，给出疾病诊断、就诊科室、检查项目、治疗方案和生活建议等，让你一目了然。它还能根据历史对话和新症状实时调整建议，推荐科室和检查项目，生成详细诊断报告和实用生活建议，让你清楚自身健康状况。对医生也是好帮手，可提升诊断和治疗效率与准确性。智能诊疗助手，时刻守护你健康，让你远离疾病，畅享美好生活！

<2 运行要求及命令：
   文件：main.py 4_diseases_infoDXYS.txt 以及yuan大模型
   命令：
	`pip install pypdf faiss-gpu langchain langchain_community langchain_huggingface streamlit==1.24.0 pillow pytesseract 
 	`streamlit run 智能诊疗助手5_报告分析new.py --server.address 127.0.0.1 --server.port 6006

<3 测试问答
	例：80贫血
	症状：最近皮肤和嘴唇颜色苍白，经常头晕、失眠，感觉记忆力也变差了

<4 数据说明：
1_MedicalQ_A.txt(1_2MedicalQ_A去重版) 2000条医疗问答数据，中文，结构为提问（症状描述）+回答（疑似病症+医疗建议），源数据来自S. Zhang， X. Zhang， H. Wang， L. Guo and S. Liu， “Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection”， in IEEE Access， vol. 6， pp. 74061-74071， 2018， doi： 10.1109/ACCESS.2018.2883637. 关键词： {生物医学影像;数据挖掘;语义学;医疗服务;特征提取;知识发现;医疗问答;互动关注;深度学习;深度神经网络}，经过处理，处理方法见3源大模型RAG实战.ipynb

2_DiseasesData.txt 400条疾病数据，英文由百度翻译为中文，结构为疾病名+症状+治疗手段，数据来自开源数据集Hugging Face QuyenAnhDE/Diseases_Symptoms

3_output-prompt.jsonl 124条医疗问答数据，英文由百度翻译为中文，结构为prompt+症状+可能诊断，数据来自开源数据集Hugging Face varshil27/Symtoms-Disease-LLama2-Format

4_diseases_infoDXYS.txt 近200条疾病相关完整数据，来自丁香医生相关页面，网址链接https://dxy.com/diseases/，数据经过处理
