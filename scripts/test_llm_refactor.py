import asyncio
import os
import sys
from pydantic import BaseModel
from typing import List

# Adjust path to find src
sys.path.append(os.getcwd())

from src.services.llm_service import LLMService

class Person(BaseModel):
    name: str
    age: int
    email: str

async def main():
    service = LLMService()
    print("Initialized LLMService")
    
    # 1. Test Simple Chat (OpenAI or Google if available)
    print("\n--- Testing Chat Generation ---")
    if service.openai_key:
        print("Testing OpenAI Chat...")
        try:
            res = await service.generate_response("Say hello!", provider="openai")
            print(f"OpenAI Response: {res[:50]}...")
        except Exception as e:
            print(f"OpenAI Failed: {e}")
            
    if service.google_key:
        print("Testing Google Chat...")
        try:
            res = await service.generate_response("Say hello!", provider="google")
            print(f"Google Response: {res[:50]}...")
        except Exception as e:
            print(f"Google Failed: {e}")

    # 2. Test Structured Output
    print("\n--- Testing Structured Output ---")
    if service.openai_key:
        print("Testing OpenAI Structured...")
        try:
            p = await service.generate_structured_response(
                "Extract info: Alice is 25, email alice@example.com", 
                Person, 
                provider="openai"
            )
            print(f"OpenAI Parsed: {p}")
        except Exception as e:
            print(f"OpenAI Structured Failed: {e}")

    if service.google_key:
        print("Testing Google Structured...")
        try:
            p = await service.generate_structured_response(
                "Extract info: Bob is 30, email bob@example.com", 
                Person, 
                provider="google"
            )
            print(f"Google Parsed: {p}")
        except Exception as e:
            print(f"Google Structured Failed: {e}")
            
    # 3. Test Embeddings
    print("\n--- Testing Embeddings ---")
    texts = ["hello world", "graph rag is cool"]
    
    if service.voyage_key:
        print("Testing Voyage Embeddings...")
        try:
            embs = await service.generate_embeddings_batch(texts, model_name="voyage-3-large")
            print(f"Voyage Embeddings: {len(embs)} vectors, dim {len(embs[0])}")
        except Exception as e:
            print(f"Voyage Failed: {e}")

    if service.openai_key:
        print("Testing OpenAI Embeddings...")
        try:
            embs = await service.generate_embeddings_batch(texts, model_name="text-embedding-3-large")
            print(f"OpenAI Embeddings: {len(embs)} vectors, dim {len(embs[0])}")
        except Exception as e:
            print(f"OpenAI Failed: {e}")

    if service.google_key:
        print("Testing Google Embeddings...")
        try:
            embs = await service.generate_embeddings_batch(texts, model_name="google")
            print(f"Google Embeddings: {len(embs)} vectors, dim {len(embs[0])}")
        except Exception as e:
            print(f"Google Failed: {e}")

    print("\n--- Done ---")

if __name__ == "__main__":
    asyncio.run(main())
