'''
nohup python3 a.py > log 2>&1 &
'''
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import more_itertools


def threaded_map(func, iterable,
    max_workers: int = 2,
    partition_size: int = 50
):
    if max_workers == 1:
        return [func(item) for item in iterable]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for sub_iterable in more_itertools.chunked(iterable, partition_size):
            futures = [executor.submit(func, item) for item in sub_iterable]
            for future in tqdm(as_completed(futures), total=len(sub_iterable)):
                results.append(future.result())
        return results
    

def send_request(
    prompt,
    ip: str = "localhost",
    port: str = "8000",
    url: str = "http://localhost:port/v1/completions",
    param_gen = {},
    timeout_seconds = 300,
):
    url = url.replace("localhost", str(ip))
    url = url.replace("port", str(port))
    payload = {
        "prompt": prompt,
        **param_gen
    }
    
    with requests.post(url, json=payload, timeout=timeout_seconds) as response:
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"[vllm_utils:send_request] Error: {repr(e)} - {response}")
            response = repr(e)
            finish_reason = "stop"
            return response, [finish_reason]
    
        result = response.json()
        response = result['choices'][-1]['text']
        finish_reason = result['choices'][-1]['finish_reason']
        return response, [finish_reason]


def get_model_output(
    prompt,
    param_gen,
    ip: str = "localhost",
    port: str = "8000",
    url: str = "http://localhost:port/v1/completions",
    timeout_seconds: int = 300,
):
    try:
        res, finish_reason = send_request(
            ip=ip,
            port=port,
            prompt=prompt,
            param_gen=param_gen,
            url=url,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        print(f"HTTP exception error: {error}")
        res, finish_reason = "", ["error"]
    return res, finish_reason


if __name__ == '__main__':
    from functools import partial
    ip = "10.2.4.24" # head node IP
    port = "8000"
    param_gen = {"max_tokens": 1024}
    prompt = ["你是谁" for _ in range(10)]
    # prompt = ["<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n" for _ in range(10)]
    results = threaded_map(
        partial(get_model_output, ip=ip, port=port, param_gen=param_gen),
        iterable=prompt,
        max_workers=84, # 每个8卡node并发大致28K，假设每个sample 1K，那么3 nodes支持并发84
        partition_size=128, # 多少个回传一次
    )
    print(results)
