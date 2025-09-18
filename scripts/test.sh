curl -N -X POST "http://0.0.0.0:8100/proxy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer polydevs_testkey" \
  -d '{
    "provider": "aistudio",
    "model": "gemini-2.5-flash-lite",
    "messages": [
      {"role": "user", "content": "code cho tôi centroid của cây với c++"}
    ],
    "config": {
      "thinking_budget": 24000,
      "temperature": 1.2,
      "top_p": 0.95,
      "tools": true
    },
    "system_instruction": [
      "Bạn tên là Ryuuko, trả lời ngắn gọn"
    ]
  }'
