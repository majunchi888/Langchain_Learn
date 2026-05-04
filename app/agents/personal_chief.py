# %% [markdown]
# agent开发：
# 1，定义模型（识别图片需要多模态模型，不能用deepseek）
# 2，定义工具
# 3，添加记忆管理 (LangGraph自动托管agent记忆，不用自己添加checkpoint了；但如果想用sqlite存储，也很简单，见下面注释掉的代码)
# 4，定义agent
# 5, 测试(LangGraph自带Restful API的接口，定义好agnet就可以，agent.invoke会自动处理多模态输入，不用自己写代码解析了；但如果想自己构造多模态消息，也很简单，见下面注释掉的代码)
# 6, 流式调用
# 7，获取会话历史
# 8，清空会话历史

# %%
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.common.logger import logger
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Load environment variables from .env file
load_dotenv(override=True)

# llm_openai = ChatOpenAI(
#     model="qwen3.5-plus",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.environ.get('DASHSCOPE_API_KEY'),
#     temperature=0,
    
#     # 关键修复参数
#     max_retries=8,                    # 增加重试次数
#     request_timeout=90,               # 提高超时时间（秒）
#     http_client=None,                 # 让它使用默认 httpx，但我们下面可以自定义
# )
from langchain.chat_models import init_chat_model

# 本笔记本含「食材图」识图；请用支持视觉的模型（如 qwen-vl-plus）。纯文本可把下面改回 qwen3.5-plus。
model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get('DASHSCOPE_API_KEY'),
    temperature=0,
)
# print(model.invoke("hello").content) # model.invoke("hello").conte/n/

# %%
##定义工具
from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=3,
    topic="general"
)

#初始化checkpointer
# 连接sqlite
# 自动获取项目正确路径，永远不报错
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "db", "personal_chief.db")

# 确保文件夹存在
os.makedirs(os.path.dirname(db_path), exist_ok=True)

connection = sqlite3.connect(db_path, check_same_thread=False)
# 初始化checkpointer
checkpointer = SqliteSaver(connection)
# 自动建表
checkpointer.setup()

# %%
##添加记忆管理（sqlite）
# from pathlib import Path
# from langgraph.checkpoint.sqlite import SqliteSaver
# import sqlite3

# 独立 checkpoint 文件，避免与其它练习共用 personal_chief.db 时读到损坏的多模态消息
# （历史里若出现 image_url.urls 会导致 DashScope 400：要求 image_url.url）
# _db_dir = Path("SalesDB")
# _db_dir.mkdir(parents=True, exist_ok=True)
# connection = sqlite3.connect(str(_db_dir / "实战_agent_checkpoint.db"), check_same_thread=False)
# checkpointer = SqliteSaver(connection)
# checkpointer.setup()

# %%
#定义agent
from langchain.agents import create_agent

system_prompt = """
你是一名专业的私人厨师。收到用户提供的食材照片或食材清单后，请严格按照以下流程执行操作：

1. 识别和评估食材：
   若用户提供食材照片，请先准确识别所有可见食材；若用户提供食材清单，则直接读取。基于食材的外观/状态，评估其新鲜度与可用量，整理出一份清晰的「当前可用食材清单」。

2. 智能食谱检索：
   优先调用 web_search 工具，以「可用食材清单」为核心关键词，搜索可直接使用这些食材制作的可行食谱。

3. 多维度评估与排序：
   对检索到的候选食谱，从「营养价值」和「制作难度」两个维度进行量化打分，优先选择「制作简单且营养丰富」的食谱，并按得分从高到低排序。

4. 结构化方案输出：
   将排序后的食谱整理为一份结构清晰的建议报告，需包含：食谱名称、核心食材、制作步骤、得分（营养/难度）、推荐理由、参考图片链接，帮助用户快速决策。

请严格遵循流程：必须优先调用 web_search 工具搜索食谱，仅当搜索不到有效结果时，才能基于食材清单自主推荐合理食谱。
"""

agent = create_agent(model=model,tools=[web_search],system_prompt=system_prompt,checkpointer=checkpointer, )


# 流式对话
async def search_recipes(prompt: str, image: str, thread_id: str):
    """调用agent搜索食谱"""
    logger.info(f"[用户]: {prompt}, image: {image}, thread_id: {thread_id}")
    try:
        # 判断是否有图片，封装不同格式的消息
        if not image or image.strip() == "":
            message = HumanMessage(content=prompt)
        else:
            message = HumanMessage(content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ])

        # 流式调用Agent
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"\n[错误]: {str(e)}")
        yield "信息检索失败，试试看手动输入食物列表？"

# 清空会话
def clear_messages(thread_id: str):
    """清空会话"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)

# 查询会话历史
def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取会话历史"""
    logger.info(f"获取历史消息，thread_id: {thread_id}")

    # 根据 thread_id 查询 checkpoint
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})

    # 如果不存在，返回空列表
    if not checkpoint:
        return []

    # 安全获取 messages
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    # 转换消息格式
    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result


# %%
# # 测试
# import uuid

# multimodal_message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "帮我看看这张图片能做什么",
#         },
#         {
#             "type": "image_url",
#             "image_url": {"url": "https://picsum.photos/id/292/400/300.jpg"},
#         },
#     ]
# )
# # 新 thread_id，避免复用旧 checkpoint 里已损坏的多模态历史；要延续同一会话则改用固定字符串
# config = {"configurable": {"thread_id": f"chef-img-{uuid.uuid4().hex[:12]}"}}

# # %%
# response = agent.invoke({"messages": [multimodal_message]}, config)


# # %%
# for message in response['messages']:
#  message.pretty_print()

# # %%
# response1=agent.invoke({"messages":HumanMessage(content="我喜欢第二道菜，可以说的详细一点吗")},config)

# # %%
# print(response1)


