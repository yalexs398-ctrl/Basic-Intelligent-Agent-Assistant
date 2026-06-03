# agent.py
# agent.py
import os
import logging
from dotenv import load_dotenv

# LangChain 核心
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

from typing import Optional, List
import requests
from tavily import TavilyClient

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 验证必需的环境变量
required_env_vars = ["BAIDU_API_KEY", "BAIDU_SECRET_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"缺少必需的环境变量: {', '.join(missing_vars)}\n"
                     f"请在 .env 文件中设置这些变量")

# 1. 自定义百度千帆LLM封装 
class BaiduQianfanLLM(LLM):
    """封装百度千帆API为LangChain标准接口"""
    model: str = "ernie-3.5-turbo"

    @property
    def _llm_type(self) -> str:
        return "baidu_qianfan"

    @property
    def _identifying_params(self) -> dict:
        """返回模型参数"""
        return {"model": self.model}
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # 获取access token
            token_url = "https://aip.baidubce.com/oauth/2.0/token"
            token_params = {
                "grant_type": "client_credentials",
                "client_id": os.getenv("BAIDU_API_KEY"),     
                "client_secret": os.getenv("BAIDU_SECRET_KEY") 
            }
            token_response = requests.post(token_url, params=token_params, timeout=30)
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")
            
            if not access_token:
                raise ValueError("获取 access_token 失败")
            
            # 调用千帆API
            api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
            headers = {"Content-Type": "application/json"}
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{api_url}?access_token={access_token}",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json().get("result")
            if not result:
                error_msg = response.json().get("error_msg", "未知错误")
                raise ValueError(f"API返回错误: {error_msg}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("请求超时")
            return "抱歉，请求超时，请稍后重试。"
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {str(e)}")
            return f"网络连接失败: {str(e)}"
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return f"模型调用失败: {str(e)}"


# 2. 构建向量数据库（长期记忆）
class VectorMemory:
    """使用Chroma构建持久化知识库"""
    
    def __init__(self, persist_directory="./chroma_db"):
        try:
            # 初始化嵌入模型
            self.embeddings = QianfanEmbeddingsEndpoint(
                qianfan_ak=os.getenv("BAIDU_API_KEY"),
                qianfan_sk=os.getenv("BAIDU_SECRET_KEY"),
            )
            
            # 创建持久化向量数据库
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="agent_memory"
            )
            logger.info(f"向量数据库初始化成功，存储路径: {persist_directory}")
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            raise
        
    def add_knowledge(self, texts: List[str], metadatas: List[dict] = None):
        """向知识库添加文档"""
        if not texts:
            return
            
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}", "timestamp": "2026-06-03"} for i in range(len(texts))]
        
        try:
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            self.vector_store.persist()
            logger.info(f"成功添加 {len(texts)} 条知识到向量数据库")
        except Exception as e:
            logger.error(f"添加知识失败: {str(e)}")
        
    def search(self, query: str, k: int = 3) -> List[str]:
        """语义检索相关文档"""
        if not query.strip():
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []


# 3. 定义Agent工具集
def web_search_tool(query: str) -> str:
    """使用Tavily进行真实网络搜索"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "⚠️ 未配置 TAVILY_API_KEY，请先在 .env 文件中添加"
    
    try:
        client = TavilyClient(api_key=tavily_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        
        # 格式化搜索结果
        results = []
        if response.get('answer'):
            results.append(f"📝 AI摘要: {response['answer']}")
        
        for result in response.get('results', [])[:3]:
            results.append(f"🔗 {result['title']}: {result['snippet']}")
        
        return "\n".join(results) if results else "未找到相关信息"
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        return f"搜索失败: {str(e)}"


# 注意：这个工具将在 AdvancedAgent 初始化后动态创建
def create_vector_memory_tool(vector_memory_instance):
    """创建向量记忆工具（解决作用域问题）"""
    def search_vector_memory(query: str) -> str:
        results = vector_memory_instance.search(query, k=3)
        if results:
            return "\n".join(results)
        return "未找到相关历史记录"
    
    return Tool(
        name="VectorMemory",
        func=search_vector_memory,
        description="访问长期知识库，用于查询历史对话记录和已存储的专业知识"
    )


# 4. 构建主Agent类 
class AdvancedAgent:
    """具备记忆和工具调用能力的Agent"""
    
    def __init__(self):
        logger.info("正在初始化 AdvancedAgent...")
        
        # 初始化组件
        self.llm = BaiduQianfanLLM()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.vector_memory = VectorMemory()
        
        # 创建工具（传入 vector_memory 实例）
        self.tools = [
            Tool(
                name="WebSearch",
                func=web_search_tool,
                description="用于搜索互联网实时信息，当需要最新新闻、事实数据或我不知道的内容时使用。输入：搜索关键词"
            ),
            create_vector_memory_tool(self.vector_memory)
        ]
        
        # 初始化Agent
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,  # 设为 False 可减少输出
                max_iterations=3,
                early_stopping_method="generate",
                handle_parsing_errors=True  # 处理解析错误
            )
            logger.info("Agent 初始化成功")
        except Exception as e:
            logger.error(f"Agent 初始化失败: {str(e)}")
            raise
        
    def chat(self, user_input: str) -> str:
        """对话主入口"""
        if not user_input or not user_input.strip():
            return "请问您有什么问题？"
        
        try:
            logger.info(f"用户输入: {user_input[:50]}...")
            
            # 先检索相关历史记忆
            relevant_memories = self.vector_memory.search(user_input, k=2)
            enhanced_input = user_input
            if relevant_memories:
                context = "\n相关历史记录：\n" + "\n".join(relevant_memories)
                enhanced_input = f"{user_input}\n\n{context}"
                logger.info(f"检索到 {len(relevant_memories)} 条相关记忆")
            
            # Agent智能处理
            response = self.agent.run(input=enhanced_input)
            
            # 自动保存重要对话到长期记忆
            self._save_to_memory(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"对话处理出错: {str(e)}")
            return f"处理出错: {str(e)}"
    
    def _save_to_memory(self, question: str, answer: str):
        """保存重要对话到向量数据库"""
        # 判断是否需要保存
        if len(answer) > 50 or any(kw in answer for kw in ["根据", "搜索结果显示", "找到"]):
            knowledge_text = f"Q: {question}\nA: {answer}"
            self.vector_memory.add_knowledge(
                texts=[knowledge_text],
                metadatas=[{"type": "conversation", "timestamp": "2026-06-03"}]
            )
            logger.info("已保存对话到长期记忆")
    
    def add_documents(self, file_paths: List[str]):
        """批量导入文档构建知识库"""
        texts = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)
                logger.info(f"已读取文件: {file_path}")
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {str(e)}")
        
        if texts:
            self.vector_memory.add_knowledge(texts)
            print(f"已添加 {len(texts)} 个文档到知识库")


# 5. 运行Demo
if __name__ == "__main__":
    print("企业级 AI Agent 启动中...")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv("TAVILY_API_KEY"):
        print(" 提示: 未设置 TAVILY_API_KEY，联网搜索功能将不可用")
        print(" 如需使用搜索功能，请在 .env 中添加 TAVILY_API_KEY=你的密钥\n")
    
    try:
        # 创建Agent实例
        agent = AdvancedAgent()
        print(" Agent 启动成功！")
        print(" 输入 'exit' 或 '退出' 结束对话\n")
        
        # 对话循环
        while True:
            user_input = input(" 你: ")
            if user_input.lower() in ['exit', 'quit', '退出']:
                print(" 再见！")
                break
            
            response = agent.chat(user_input)
            print(f"\n AI助手: {response}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n\n 程序已中断，再见！")
    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")
        print(f"程序运行失败: {str(e)}")
