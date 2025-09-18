curl -X POST "http://0.0.0.0:8100/proxy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer polydevs_testkey1" \
  -d '{
    "provider": "aistudio",
    "model": "gemini-2.5-flash-lite",
    "messages": [
      {"role": "user", "content": "1 + 1 = ?"}
    ]
  }'