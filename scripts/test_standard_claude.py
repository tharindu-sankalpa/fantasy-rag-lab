
from anthropic import AnthropicVertex
import os

# Ensure we don't accidentally use the bad env var if it leaked
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

print(f"Project: {os.popen('gcloud config get-value project').read().strip()}")
print(f"Account: {os.popen('gcloud config get-value account').read().strip()}")

client = AnthropicVertex(region="global", project_id="keen-jigsaw-484410-t4")

print("Testing Claude 3.5 Sonnet (global)...")
try:
    message = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        model="claude-3-5-sonnet@20240620"
    )
    print("Success!")
    print(message.content[0].text)
except Exception as e:
    print(f"Error: {e}")
