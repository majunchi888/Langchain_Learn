from fastapi import APIRouter
from app.models.schemas import ChatRequest
from app.agents.personal_chief import search_recipes, clear_messages, get_messages
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/chat/stream") # 定义流式对话POST接口
async def chat_endpoint(request: ChatRequest): #request: ChatRequest：前端传过来的参数，固定格式
    """流式对话"""
    return StreamingResponse(
           search_recipes(request.message, request.image_url, request.thread_id),
           media_type="text/event-stream" # 告诉浏览器：这是流式数据，不是一次性 JSON，要一段段接收
           )



@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages = get_messages(thread_id)
    return {"messages": messages}



@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    clear_messages(thread_id)
    return {"success": True}