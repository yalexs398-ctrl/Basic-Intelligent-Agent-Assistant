# agent.py
import os
import logging
from typing import List
import requests
from tavily import TavilyClient

from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# 基础配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# 环境变量检查
if not os.getenv("BAIDU_API_KEY"):
    raise ValueError("请在 .env 文件中设置 BAIDU_API_KEY")

# 1. 百度千帆 LLM（V2 版本）
class BaiduQianfanLLM(LLM):
    model: str = "ernie-speed-8k"

    @property
    def _llm_type(self) -> str:
        return "baidu_qianfan"

    def _call(self, prompt: str, stop: List[str] | None = None) -> str:
        url = "https://qianfan.baidubce.com/v2/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('BAIDU_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return "模型调用失败，请稍后重试。"

# 2. 联网搜索工具（Tavily）
def web_search_tool(query: str) -> str:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "未配置 TAVILY_API_KEY，无法使用联网搜索。"

    try:
        client = TavilyClient(api_key=tavily_key)
        result = client.search(query=query, max_results=3)
        return "\n".join(
            f"{r['title']}: {r['content']}" for r in result.get("results", [])
        )
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return "搜索失败，请稍后再试。"

# 3. Agent 主体
class AdvancedAgent:
    def __init__(self):
        self.llm = BaiduQianfanLLM()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.tools = [
            Tool(
                name="WebSearch",
                func=web_search_tool,
                description="当你不知道答案或需要最新信息时，使用此工具搜索互联网。"
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )

        logger.info("✅ Agent 初始化完成")

    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return "请输入你的问题。"
        return self.agent.run(user_input)

# 4. 运行入口
if __name__ == "__main__":
    print("AI Agent 已启动")
    print("输入 exit / quit / 退出 结束对话\n")

    agent = AdvancedAgent()

    while True:
        user_input = input("你：")
        if user_input.lower() in {"exit", "quit", "退出"}:
            print(" 再见！")
            break

        reply = agent.chat(user_input)
        print(f"\nAI：{reply}\n")
