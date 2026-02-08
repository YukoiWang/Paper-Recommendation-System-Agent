"""
LLM Service
大语言模型服务，支持多种LLM提供商
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from backend.config import settings


class LLMService:
    """LLM Service for various LLM providers"""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化LLM客户端"""
        try:
            if self.provider == "openai":
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL
                )
            elif self.provider == "qwen":
                # TODO: Initialize Qwen client
                logger.warning("Qwen provider not implemented yet, using OpenAI")
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL
                )
            elif self.provider == "deepseek":
                # TODO: Initialize DeepSeek client
                logger.warning("DeepSeek provider not implemented yet, using OpenAI")
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    async def generate_profile_summary(self, user_data: dict) -> str:
        """
        生成用户画像摘要
        
        Args:
            user_data: 用户数据（阅读历史、兴趣等）
            
        Returns:
            用户画像摘要文本
        """
        prompt = f"""根据以下用户数据，生成一段简洁的用户研究兴趣画像摘要（2-3句话）：

用户数据: {user_data}

请用中文生成画像摘要，突出用户的研究方向和兴趣点。"""
        
        return await self.generate(prompt, temperature=0.5)
