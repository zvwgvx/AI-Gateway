curl -N -X POST "https://api.polydevs.uk/proxy" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer polydevs_testkey" \
  --data-binary @- <<'JSON'
{
  "provider": "polydevs",
  "model": "ryuuko-r1-eng-mini",
  "messages": [
    {"role": "user", "content": "hey ryu, wana go to drink matcha with me ?"}
  ],
  "config": {
    "thinking_budget": -1,
    "temperature": 1.2,
    "top_p": 0.85,
    "tools": true
  },
  "system_instruction": [""]
}
JSON
