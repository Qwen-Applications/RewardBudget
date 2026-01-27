# llm_handler.py
import json
import os
import random
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests
import sseclient
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException, Timeout


class LLMHandler:
    def __init__(self, 
        text_model_name: str = "qwen3-235b-instruct-trans-sg", # "qwen3-235b-instruct-trans-128k"
        vlm_model_name: str = "qwen3l-vl-235-instruct-trans4",
        coder_model_name: str = "qwen3-235b-instruct-2507-trans", # "qwen3-235b-instruct-trans-128k"
    ):
        # 初始化自定义平台的API信息
        # 允许用环境变量覆盖，便于在不同集群/网关环境下切换。
        self.text_model_url = os.environ.get(
            "QWEN_TEXT_MODEL_URL",
            "http://shenma-ai-pub.alibaba-inc.com/v1/completions",
        )
        self.vlm_model_url = os.environ.get(
            "QWEN_VLM_MODEL_URL",
            "http://shenma-ai-pub.alibaba-inc.com/v1/completions",
        )
        self.api_key = os.environ.get("QWEN_API_KEY", "REDACTED_QWEN_API_KEY")

        # API 稳定性参数（可通过环境变量调节）
        self.max_retries = int(os.environ.get("QWEN_API_MAX_RETRIES", "4"))
        self.retry_backoff_s = float(os.environ.get("QWEN_API_RETRY_BACKOFF_S", "0.5"))

        # 设置模型名称
        self.text_model_name = text_model_name
        self.coder_model_name = coder_model_name
        self.vlm_model_name = vlm_model_name

        # NOTE: evaluation.py sets `handler.verbose` when using this as LLM-as-judge.
        # Default should be silent to avoid flooding stdout with success logs.
        self.verbose = getattr(self, "verbose", False)

    def _log(self, *args, **kwargs):
        if getattr(self, "verbose", False):
            print(*args, **kwargs)

    def _is_retryable_http_status(self, code: int) -> bool:
        # 404 在某些网关/灰度环境下可能是临时路由失败；这里把它视作可重试。
        return code in {404, 408, 409, 425, 429, 500, 502, 503, 504}

    def _sleep_backoff(self, attempt: int) -> None:
        base = float(getattr(self, "retry_backoff_s", 0.5))
        # 指数退避 + 抖动，避免雪崩重试
        time.sleep(base * (2 ** attempt) + random.random() * base)

    def _preview_response_text(self, resp: requests.Response, limit: int = 800) -> str:
        """Best-effort preview of response body without assuming it is fully buffered."""
        try:
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=256):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) >= limit:
                    break
            return bytes(buf[:limit]).decode("utf-8", errors="replace")
        except Exception:
            try:
                return (resp.text or "")[:limit]
            except Exception:
                return ""


    def call_qwen3(self, messages: List[Dict[str, str]], stream: bool = True, model_name=None) -> Any:
        """
        调用Qwen3-max模型

        Args:
            messages: 对话消息列表
            stream: 是否流式输出

        Returns:
            模型响应
        """
        self._log("processing text_model_name:", self.text_model_name)
        model_name = model_name or self.text_model_name
        prompt = self._convert_openai_messages_to_prompt(messages)
        request_data = self._bulid_request_data(prompt, model=model_name)

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # NOTE:
        # - 该后端返回的是 text/event-stream (SSE)。
        # - 即使调用方 stream=False，我们也必须用 stream=True 发起请求，才能稳定消费 SSE chunk。
        timeout = getattr(self, "timeout", None)
        max_retries = int(getattr(self, "max_retries", 4))

        last_err: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.text_model_url,
                    json=request_data,
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                )
            except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as e:
                last_err = e
            else:
                if not response.ok:
                    body_preview = self._preview_response_text(response, 800)
                    err = RuntimeError(
                        f"qwen_primus HTTP {response.status_code}; "
                        f"url={self.text_model_url}; model={model_name}; "
                        f"request_id={request_data.get('request_id')}; "
                        f"body[:800]={body_preview!r}"
                    )
                    # 某些网关/灰度下 404/5xx 可能是暂态；允许重试。
                    if self._is_retryable_http_status(response.status_code) and attempt < max_retries - 1:
                        last_err = err
                        try:
                            response.close()
                        except Exception:
                            pass
                        self._sleep_backoff(attempt)
                        continue
                    raise err

                client = sseclient.SSEClient(response)

                if stream:
                    # 流式：返回 generator，由调用方消费；不要提前关闭 response。
                    return self._adapt_sse_to_openai_stream(client)

                # 非流式：消费到 complete 事件后返回最终文本。
                try:
                    for event in client.events():
                        if event.event == "complete":
                            raw = json.loads(event.data)
                            return raw["choices"][0]["message"]["content"]
                    last_err = RuntimeError("qwen_primus SSE stream ended without 'complete' event")
                except (ChunkedEncodingError, ConnectionError, Timeout, RequestException, json.JSONDecodeError, ValueError) as e:
                    last_err = e
                finally:
                    try:
                        response.close()
                    except Exception:
                        pass

            if attempt < max_retries - 1:
                self._log("qwen_primus retrying after error:", repr(last_err))
                self._sleep_backoff(attempt)
                continue
            break

        raise RuntimeError(
            f"qwen_primus request failed after {max_retries} attempts; "
            f"url={self.text_model_url}; model={model_name}; "
            f"request_id={request_data.get('request_id')}; "
            f"last_err={type(last_err).__name__ if last_err else None}: {last_err}"
        )

    def _adapt_sse_to_openai_stream(self, sse_client):
        """
        将内部平台 SSEClient 的事件流包装成类似 OpenAI stream 的chunk，便于调用端统一处理。
        """
        last_content = ""  # 记录上一次的完整字符串
        # 记录耗时
        start_time = time.time()

        for event in sse_client.events():        
            try:
                raw = json.loads(event.data)
            except Exception as e:
                data_preview = (getattr(event, "data", "") or "")[:800]
                raise RuntimeError(f"qwen_primus SSE json decode failed: {type(e).__name__}: {e}; data[:800]={data_preview!r}")
            
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

            if event.event == "complete":
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
        self._log(request_id)
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
    llm_handler = LLMHandler(text_model_name="qwen3-235b-instruct-trans-sg")

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
