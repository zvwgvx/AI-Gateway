curl -X POST "http://0.0.0.0:8100/proxy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer polydevs_testkey" \
  -d '{
    "provider": "aistudio",
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "Xin chào"}
    ]
  }'