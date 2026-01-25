from anthropic import AnthropicVertex

client = AnthropicVertex(region="asia-east1", project_id="keen-jigsaw-484410-t4")

message = client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello! Can you help me?"}],
    model="claude-sonnet-4@20250514"
)
print(message.content[0].text)