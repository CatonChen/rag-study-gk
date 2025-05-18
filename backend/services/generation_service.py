from typing import List, Dict, Any, Optional
import openai
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class GenerationService:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        results_dir: str = "05-generation-results"
    ):
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置 OpenAI API 密钥
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """生成回答"""
        # 构建提示词
        context_text = "\n\n".join([
            f"文档 {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context)
        ])
        
        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文信息：
{context_text}

问题：{query}

回答："""
        
        try:
            # 调用 OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的助手，基于提供的上下文信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取回答
            answer = response.choices[0].message.content
            
            # 构建结果
            result = {
                "query": query,
                "answer": answer,
                "context": context,
                "model": self.model_name,
                "usage": response.usage
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "context": context
            }
    
    def save_result(self, result: Dict[str, Any], output_file: str):
        """保存生成结果"""
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def load_result(self, file_path: str) -> Dict[str, Any]:
        """加载生成结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            models = openai.Model.list()
            return [model.id for model in models.data]
        except Exception as e:
            return ["gpt-3.5-turbo", "gpt-4"]  # 返回默认模型列表
    
    def stream_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """流式生成回答"""
        # 构建提示词
        context_text = "\n\n".join([
            f"文档 {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context)
        ])
        
        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文信息：
{context_text}

问题：{query}

回答："""
        
        try:
            # 调用 OpenAI API 进行流式生成
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的助手，基于提供的上下文信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # 返回生成器
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}" 