import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from typing import Any, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import pytesseract

# 向量模型下载
from modelscope import snapshot_download
embedding_model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='./')

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
#model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
#model_path = './IEITYuan/Yuan2-2B-July-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-large-zh-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16

# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2_LLM
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        super().__init__()
        
        try:        # 加载预训练的分词器和模型
            print("Creat tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
            self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

            print("Creat model...")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=4096)
        output = self.tokenizer.decode(outputs[0])
        #output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output.split("<sep>")[-1].split("<eod>")[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

# 定义一个函数，用于获取llm和embeddings
@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # 设置为True以计算余弦相似度
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings

#2. 向患者提问以获取更多有助于诊断的信息：例如症状的开始时间、持续时间、疼痛程度、是否有其他伴随症状、既往病史、家族病史等。 #：包括疾病的常见症状、可能的病因。，并说明每种治疗方案的优缺点。    请确保涵盖以下几点：
# ——————————————————————检查报告分析————————————————————————
Analyze_template = """
假设你是一个检验医生，请根据以下检查报告内容，分以下几点给出一些初步的诊断建议：
1. 最可能的两种疾病诊断及诊断理由，简单列出每种疾病的常见症状。
2. 需要进一步检查的项目。
3. 简单解释可能的治疗方案。
4. 具体的生活方式或饮食方面的建议。

    检查报告内容如下：
    {report_text}
"""
# 诊断的prompt模板
diagnose_prompt_template = """
假设您是一位经验丰富、尊重和诚实的医生助理。请基于患者情况和背景信息，完成以下任务：
1. 给出最可能的疾病诊断及诊断理由，简单列出每种疾病的常见症状、可能的病因。
2. 给出适合的就诊科室，建议需要进行的检查项目。
3. 简单解释可能的治疗方案，
4. 给出具体的生活建议，以帮助患者更好地管理健康状况。

患者情况和背景信息如下: 
{situation_and_context}

如果信息不足，请向患者提问以获取更多有助于诊断的信息：例如症状的持续时间、疼痛程度、是否有其他伴随症状、既往病史、家族病史等。
"""

# 总结诊断和建议的prompt模板
conclude_prompt_template = """
假设您是一位乐于助人、可靠、尊重和诚实的医生助理。请基于对话记录完成以下任务：
1. 明确给出最可能的疾病诊断。
2. 针对诊断，建议需要进行的具体检查项目。
3. 提供具体可行的治疗方案，非药物治疗的详细步骤。
4. 给出饮食、运动、作息等方面的生活建议，简单说明每项建议对疾病管理的作用。

对话记录: {conversation}
"""

class IntelligentDiagnosticAssistant:
    # 定义IntelligentDiagnosticAssistant类
    def __init__(self, llm, embeddings,file_path):
        #1. 导入参数&调用方法初始化
        self.llm = llm
        self.embeddings = embeddings
        self.file_path = file_path
        self.memory = ConversationBufferMemory(memory_key="conversation", return_messages=True) #涉及对话历史
        
        #2. 加载 text_splitter 用于将文本切分成较小的块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,#,100
            chunk_overlap=10,#5
            length_function=len
        )
        
        #3. 给出不同prompt
        self.analyze_prompt = PromptTemplate(input_variables= ["report_text"], template=Analyze_template)
        self.diagnose_prompt = PromptTemplate(input_variables=["situation_and_context"], template=diagnose_prompt_template)
        self.conclude_prompt = PromptTemplate(input_variables=["conversation"], template=conclude_prompt_template)
        
        #4. 用于加载问答链
        self.analyze_chain = LLMChain(llm=self.llm, prompt=self.analyze_prompt)
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=self.diagnose_prompt, memory=self.memory)
        self.diagnosis_conclu_chain = LLMChain(llm=self.llm, prompt=self.conclude_prompt)
        
    #5. 定义embedding函数，将医疗语料库转成向量
    def vectors(self,file_path):
        # 读取文件内容，将每一行合并为一个单一字符串
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.documents = file.read()  # 将整个文件内容作为单一字符串
        # 切分成chunks
        all_chunks = self.text_splitter.split_text(text=self.documents)
        # 转成向量并存储
        vector_store = FAISS.from_texts(all_chunks, embedding=self.embeddings)
        return vector_store  # 添加返回，不需要重复加载
    
    #6. 检索相似的 chunks
    def get_context(self,vector_store,symptoms): 
        retrieved_chunks = vector_store.similarity_search(query=symptoms, k=2)
        return retrieved_chunks
    
    #7. 依据情况生成回复    
        # 检查报告分析
    def analyze_report(self, docs):
        results = self.analyze_chain.run({"report_text": docs})
        return results
    
    def get_diagnosis(self, dialog_history, context):
        # 合并对话历史和上下文为一个字符串
        combined_input = "<n>".join([f"{msg['role']}: {msg['content']}" for msg in dialog_history])
        combined_input += f"<n> 背景信息: {context}"
        result = self.diagnosis_chain.run({"situation_and_context": combined_input})
        return result

    def diagnosis_conclu(self, dialog_history ):
        # 从记忆中获取对话内容
        conversation = "<n>".join([f"{msg['role']}: {msg['content']}" for msg in dialog_history])
        # 确保 conversation 不是空的
        if not conversation:
            raise ValueError("生成诊疗建议时出现错误: 对话记录为空或未加载正确。")
        # 生成最终诊断
        result = self.diagnosis_conclu_chain.run({"conversation": conversation})
        return result


def main():
        
    # 创建一个标题
    st.title('👩‍⚕️ 智能健康小助手')
    # 获取llm和embeddings
    if 'llm' not in st.session_state or 'embeddings' not in st.session_state:
        try:
            st.session_state.llm, st.session_state.embeddings = get_models()
        except Exception as e:
            st.error(f"加载模型时出现错误: {e}")
            return
    #file_path = "./1_2MedicalQ_A.txt" #2_DiseasesData.txt
    file_path = "./4_diseases_infoDXYS.txt"
        
    # 初始化智能诊断助手
    assistant = IntelligentDiagnosticAssistant(
        st.session_state.llm,
        st.session_state.embeddings,
        file_path
    )

    if 'dialog_history' not in st.session_state:
        st.session_state.dialog_history = []
        
    if 'vectors' not in st.session_state:
        try:
            with st.spinner('正在加载知识库，请稍候...'):
                st.session_state.vectors = assistant.vectors(file_path)
                st.success('知识库加载成功！')
        except Exception as e:
            st.error(f"初始化向量存储时出现错误: {e}")
            return 
     #——————————————————————诊断报告——————————————————————
    uploaded_file = st.file_uploader("上传检查报告", type=["pdf", "png", "jpg", "jpeg"], help="上传您的检查报告文件（PDF或图片）")
    if uploaded_file :
        file_type = uploaded_file.type
        file_content = uploaded_file.read()
        
        if file_type == "application/pdf":
            st.write("您上传了一个PDF文件")
            # 使用临时文件保存上传的PDF文件
            temp_file_path = "temp.pdf"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            # 加载临时文件中的内容
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            # 提取内容
            content = ""
            for doc in docs:
                content += doc.page_content  # 假设每个 doc 对象有 page_content 属性
        else:
            st.write("您上传了一个图片文件")
            image = Image.open(uploaded_file)
            content = pytesseract.image_to_string(image)
        
        # 生成概括
        Analysis = assistant.analyze_report(content)
        st.write(Analysis)

    # 聊天记录展示
    with st.container():
        for message in st.session_state.dialog_history:
            st.chat_message(message['role']).write(message['content'])

    # 诊断,将用户输入添加到对话历史，并在聊天界面上显示
    if user_input:= st.text_input("我是您的专属健康助手，请描述您的症状或不适，我将提供健康建议", help="输入后按 Enter 键提交。"):
        st.session_state.dialog_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(f"正在分析您的情况，为您提供专属建议，请稍候...")        #st.spinner('正在分析您的症状，请稍候...')

        try: #用对话历史和context进行分析
            # 切割对话历史
            max_history_length = 6  # 保留最近6条对话
            relevant_history = st.session_state.dialog_history[-max_history_length:]
            
            #获取背景知识
            context = assistant.get_context(st.session_state.vectors, user_input)

            # 生成诊断
            diagnosis = assistant.get_diagnosis(relevant_history, context)
            
            st.session_state.dialog_history.append({"role": "assistant", "content": diagnosis})
            st.chat_message("assistant").write(diagnosis)
       
        except Exception as e:
            st.error(f"处理诊断时出现错误: {e}")

    # 侧边栏添加“生成综合诊疗建议”
    with st.sidebar:
        if st.button("生成综合诊疗建议"):
            st.chat_message("assistant").write('正在生成综合诊疗建议，请稍候...')
            #st.spinner('正在生成综合诊疗建议，请稍候...')
            try:
                diagnosis_conclusion = assistant.diagnosis_conclu(st.session_state.dialog_history)
                st.sidebar.write("综合诊疗建议:")
                st.sidebar.write(diagnosis_conclusion)
                st.sidebar.write("以上建议仅作为参考，如果症状持续或加重，请咨询医生。")
            except ValueError as e:
                st.error(f"生成综合诊疗建议时出现错误: {e}")

if __name__ == '__main__':
    main()
    
# 删去初次对话的差别，直接用历史资料+context拼接