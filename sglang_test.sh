# 启动服务
python3 -m sglang.launch_server \
        --model-path /pfs-verdent/libaoguo/models/Qwen_Qwen3-Reranker-8B \
        --trust-remote-code \
        --disable-radix-cache \
        --host 0.0.0.0 \
        --port 8000
# 发送请求
curl -X POST "http://45.78.195.171:8998/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/pfs-verdent/libaoguo/models/Qwen_Qwen3-Reranker-8B",
    "prompt": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: Explain gravity\n<Document>: Gravity is a force that attracts two bodies towards each other.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "max_tokens": 1,
    "temperature": 0,
    "return_logits": true
  }'