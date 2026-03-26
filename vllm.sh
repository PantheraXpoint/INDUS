# conda activate vllm
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --max-model-len 16384 \
#   --swap-space 1 \
#   --gpu-memory-utilization 0.85 \
#   --dtype bfloat16 \
#   --limit-mm-per-prompt '{"image":20,"video":0}' \
#   --mm-encoder-tp-mode data \
#   --enforce-eager \
#   --trust-remote-code
lmdeploy serve api_server \
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --model-format awq \
  --backend turbomind \
  --tp 1 \
  --session-len 110000 \
  --server-port $PORT
