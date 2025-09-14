curl -X POST "https://api.polydevs.uk/proxy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer polydevs_testkey1" \
  -d '{
    "provider": "aistudio",
    "model": "gemini-2.5-flash-lite",
    "messages": [
      {"role": "user", "content": "1 + 1 = ?"}
    ]
  }'