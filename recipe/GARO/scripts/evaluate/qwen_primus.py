"""Qwen/Primus LLM API handler.

The upstream endpoint responds with Server-Sent Events (SSE). The original
version depended on an external `sseclient` package, which may not be
installed in all environments. This module now includes a small SSE parser
so it works out-of-the-box.

This client is intentionally lightweight and is used by the evaluation script
as an LLM-as-a-judge backend.
"""

import json
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests


class LLMHandler:
    def __init__(
        self,
        text_model_name: str = "qwen3-235b-instruct-trans-sg",  # "qwen3-235b-instruct-trans-128k"
        vlm_model_name: str = "qwen3l-vl-235-instruct-trans4",
        coder_model_name: str = "qwen3-235b-instruct-2507-trans",  # "qwen3-235b-instruct-trans-128k"
        *,
        api_key: Optional[str] = None,
        text_model_url: Optional[str] = None,
        vlm_model_url: Optional[str] = None,
        verbose: bool = False,
        timeout: int = 60,
    ):
        """Create a handler.

        Args:
            api_key: API key. If None, uses the default value in the original code.
            text_model_url: Override text model endpoint.
            vlm_model_url: Override VLM endpoint.
            verbose: Whether to print request IDs / debug logs.
        """

        # 初始化自定义平台的API信息
        self.text_model_url = text_model_url or "http://shenma-ai-pub.alibaba-inc.com/v1/completions"
        self.vlm_model_url = vlm_model_url or "http://shenma-ai-pub.alibaba-inc.com/v1/completions"
        # NOTE: keep backward-compat default, but allow override via args/env.
        self.api_key = api_key or "REDACTED_QWEN_API_KEY"  # 根据实际API密钥进行修改
        self.verbose = verbose
        self.timeout = timeout

        # 设置模型名称
        self.text_model_name = text_model_name
        self.coder_model_name = coder_model_name
        self.vlm_model_name = vlm_model_name

    # -----------------
    # SSE utilities
    # -----------------
    def _iter_sse_events(self, response: requests.Response) -> Iterator[Tuple[str, str]]:
        """Iterate SSE events from a streaming response.

        Yields tuples of (event_type, data_str).
        """
        event_type: Optional[str] = None
        data_lines: List[str] = []

        # Some servers may send keep-alive empty lines.
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()

            if not line:
                # event boundary
                if data_lines:
                    yield (event_type or "message", "\n".join(data_lines))
                event_type = None
                data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
                continue

            # ignore other SSE fields like id:, retry:, etc.

        # flush if stream ends without final blank line
        if data_lines:
            yield (event_type or "message", "\n".join(data_lines))

    def call_qwen3(self, messages: List[Dict[str, str]], stream: bool = True, model_name=None) -> Any:
        """
        调用Qwen3-max模型
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            
        Returns:
            模型响应
        """
        if self.verbose:
            print("processing text_model_name:", self.text_model_name)
        model_name = model_name or self.text_model_name
        prompt = self._convert_openai_messages_to_prompt(messages)
        request_data = self._bulid_request_data(prompt, model=model_name)

        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            self.text_model_url,
            json=request_data,
            headers=headers,
            stream=True,  # endpoint is SSE; we always stream and parse
            timeout=(self.timeout, self.timeout),
        )

        response.raise_for_status()
        events = self._iter_sse_events(response)

        if stream:
            return self._adapt_sse_to_openai_stream(events)

        last_payload: Optional[Dict[str, Any]] = None
        for ev_type, ev_data in events:
            if not ev_data:
                continue
            payload = json.loads(ev_data)
            last_payload = payload
            if ev_type == "complete":
                break

        if not last_payload:
            raise RuntimeError("No SSE payload received from the server.")

        # OpenAI-compatible parsing
        return last_payload["choices"][0]["message"]["content"]

    def _adapt_sse_to_openai_stream(self, events: Iterable[Tuple[str, str]]):
        """Wrap SSE events into OpenAI-like streaming chunks.

        The calling side can iterate over the returned generator.
        """
        last_content = ""  # 记录上一次的完整字符串
        # 记录耗时
        start_time = time.time()

        for event_type, event_data in events:
            if not event_data:
                continue
            raw = json.loads(event_data)
            
            # print(raw)

            if "choices" in raw and raw["choices"]:
                full_content = raw["choices"][0]["message"].get("content", "")

                # 找出新增的部分
                if full_content.startswith(last_content):
                    new_part = full_content[len(last_content):]
                else:
                    # 如果内容改变或重置（例如Clear等），直接视为全新内容
                    new_part = full_content

                last_content = full_content  # 更新记录

                if new_part:  # 有新增才yield
                    # 补充耗时
                    raw["usage"]['time_cost'] = round(time.time() - start_time, 3)                    
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                index=raw["choices"][0].get("index", 0),
                                delta=SimpleNamespace(content=new_part),
                                finish_reason=None
                            )
                        ],
                        usage=raw["usage"]
                    )

            if event_type == "complete":
                break

    def call_qwen3_coder(self, messages: List[Dict[str, str]], stream: bool = True) -> Any:
        return self.call_qwen3(messages, stream, self.coder_model_name)

    def remove_emojis_keep_spaces(self, text):
        """移除表情符号但保持文本格式"""
        if not isinstance(text, str):
            return text
        
        # 表情符号正则模式
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002600-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "]+",
            flags=re.UNICODE
        )
        
        # 替换为空格，然后清理多余空格
        text = emoji_pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
        return text.strip()
    def _bulid_request_data(self, prompt, model="qwen3-235b-instruct-trans-sg"):
        request_id = uuid.uuid4().hex
        session_id = uuid.uuid4().hex
        if self.verbose:
            print("request_id:", request_id)
        request_data = {
            "session_id": session_id,
            "request_id": request_id,
            "model": model,
            "prompt": prompt,
            "extra_args": {
                "top_k": 50,
                "top_p": 0.8,
                "temperature": 0.5,
                "max_prefill_tokens": 65536,
                "context_length": 65536,
                "max_new_tokens": 32767,
            },
        }
        return request_data

    # def _bulid_request_data(self, prompt, model="qwen3-235b-instruct-2507-trans"):
    #     request_data = {
    #         "session_id": uuid.uuid4().hex,
    #         "request_id": uuid.uuid4().hex,
    #         "model": model,
    #         "prompt": prompt,
    #         "extra_args": {
    #             "top_k": 50,
    #             "top_p": 0.8,
    #             "temperature": 0.5,
    #             "max_tokens": 32767,
    #         },
    #     }
    #     return request_data

    def _bulid_request_data_vlm(self, image_urls, prompt, model="qwen3-235b-instruct-2507-trans"):
        request_data = {
            "session_id": uuid.uuid4().hex,
            "request_id": uuid.uuid4().hex,
            "model": model,
            "prompt": prompt,
            "multi_modal_data": [
                {"mime_type": "image/url", "content": url} for url in image_urls
            ],
            "extra_args": {
                "top_k": 50,
                "top_p": 0.8,
                "temperature": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": 8192,
            },
        }
        return request_data

    def _convert_openai_messages_to_prompt(self, messages):
        """
        将 OpenAI 格式的 messages 列表 转换为 平台 prompt 字符串

        Parameters:
            messages (list[dict]): OpenAI 格式的消息列表，每个消息形如：
                {'role': 'system', 'content': '...'}
                {'role': 'user', 'content': '...'}
                {'role': 'assistant', 'content': '...'}

        Returns:
            str: 平台使用的 prompt 字符串
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = self.remove_emojis_keep_spaces(msg["content"])
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # 按惯例，最后加上 assistant 的起始标记，表示等待模型生成回复
        parts.append("<|im_start|>assistant\n")
        
        return "".join(parts)# 使用示例


if __name__ == "__main__":
    # 测试代码
    # llm_handler = LLMHandler(text_model_name="qwen3-235-trans5-fp8")
    # llm_handler = LLMHandler(text_model_name="qwen3-235b-instruct-2507-trans")
    llm_handler = LLMHandler(text_model_name="qwen3-235b-instruct-trans-sg", verbose=True)

    # 测试Qwen3-max (非流式)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你好，你是谁？'}
    ]

    # # 读取保存的 messages
    # import numpy as np
    # messages = np.load('tmp/messages.npy', allow_pickle=True).tolist()
    # for message in messages:
    #     message['content'] = message['content']
        # with open(f'{message["role"]}.txt', 'w') as f:
        #     f.write(message['content'])

    response = llm_handler.call_qwen3(messages, stream=False)
    print(response)

    # # 测试流式输出
    # print("流式输出测试:")
    # response_stream = llm_handler.call_qwen3(messages, stream=True)
    # for chunk in response_stream:
    #     if chunk and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="", flush=True)
    # print("token/s:", chunk.usage["completion_tokens"] / chunk.usage['time_cost'])
