from collections.abc import Callable, Iterator, Sequence
from threading import Thread
from typing import Any

import torch
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

from app.infrastructure.config.settings import settings


class LocalQwen3VL(BaseChatModel):
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    model: Any = None
    processor: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config_model = settings.local_models.ocr_model
        if config_model:
            self.model_name = config_model
            
        self._load_model()
        
    def _load_model(self):
        if self.model is None:
            print(f"正在加载本地 Qwen3-VL：{self.model_name}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            try:
                from huggingface_hub import HfApi
                from tqdm.auto import tqdm

                tqdm.write(f"📦 正在下载视觉语言模型 {self.model_name}...")

                try:
                    api = HfApi()
                    repo_info = api.repo_info(self.model_name, repo_type="model")
                    siblings = getattr(repo_info, 'siblings', [])
                    total_files = len(siblings) if siblings else 30

                    with tqdm(total=total_files, desc=f"下载 {self.model_name}", unit="文件") as pbar:
                        for sibling in siblings:
                            filename = sibling.rfilename if hasattr(sibling, 'rfilename') else sibling
                            try:
                                api.hf_hub_download(
                                    filename=filename,
                                    repo_id=self.model_name,
                                    repo_type="model",
                                    resume_download=True,
                                )
                            except Exception:
                                pass
                            pbar.update(1)
                except Exception:
                    pass

                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, 
                    torch_dtype=dtype, 
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
                if device == "cpu":
                    self.model = self.model.to("cpu")
                    
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                print(f"本地 Qwen3-VL 已在 {device} 上加载完成。")
            except Exception as e:
                print(f"加载本地 Qwen3-VL 失败：{e}")
                raise e

    @property
    def _llm_type(self) -> str:
        return "local-qwen3-vl"

    def bind_tools(self, tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool], **kwargs: Any) -> Runnable:
        """
        Local Qwen 的 bind_tools 伪实现。
        """
        print("警告：LocalQwen3VL 暂不支持原生工具绑定，将忽略传入的 tools。")
        return self
    
    def _messages_to_conversation(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            role = "user"
            if isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            content: list[dict[str, Any]] = []
            if isinstance(msg.content, str):
                content.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content.append({"type": "text", "text": item.get("text")})
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url")
                            if url:
                                content.append({"type": "image", "image": url})
            conversation.append({"role": role, "content": content})
        return conversation

    def _prepare_inputs(self, conversation: list[dict[str, Any]]):
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            conversation = self._messages_to_conversation(messages)
            inputs = self._prepare_inputs(conversation)
            
            # 配置流式输出
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # 在单独线程中执行生成
            max_new_tokens = kwargs.get("max_new_tokens", 1024)
            generation_kwargs = dict(inputs, max_new_tokens=max_new_tokens, streamer=streamer)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 逐块产出结果
            for new_text in streamer:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=new_text))
                if run_manager:
                    run_manager.on_llm_new_token(new_text, chunk=chunk)
                yield chunk
                
        except Exception as e:
            print(f"流式生成失败：{e}")
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"Error: {str(e)}"))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            conversation = self._messages_to_conversation(messages)
            inputs = self._prepare_inputs(conversation)

            # 生成
            with torch.no_grad():
                max_new_tokens = kwargs.get("max_new_tokens", 1024)
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # 裁剪输入 token
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output_text))])
            
        except Exception as e:
            print(f"生成失败：{e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])
