import os
from google import genai
from dotenv import load_dotenv

# Get the path to the project root (assuming script is in scripts/ subdir)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
load_dotenv(os.path.join(project_root, ".env"))

def check_gemini_access():
    """
    Checks if the GEMINI_API_KEY has access to Gemini models,
    specifically looking for 'Gemini 3' or the latest available versions.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    print(api_key)
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env or environment variables.")
        print("   Please ensure you have a .env file with GEMINI_API_KEY set.")
        return

    # Mask key for privacy in output
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"🔑 Found GEMINI_API_KEY: {masked_key}")
    
    # Configure the client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Error configuring Gemini client: {e}")
        return

    print("\n-------- Checking Available Models --------")
    try:
        # List all models
        # Note: genai.list_models() returns a generator
        all_models = list(client.models.list())
        
        # Filter for 'generateContent' capable models (usually what we care about for chat/text)
        # and specifically look for 'gemini' models
        chat_models = [
            m for m in all_models 
            if m.supported_actions and 'generateContent' in m.supported_actions and 'gemini' in m.name
        ]
        
        # Sort for easier reading
        chat_models.sort(key=lambda x: x.name)

        if not chat_models:
            print("⚠️  No 'gemini' models found that support content generation.")
            return
        
        gemini_3_models = []
        latest_model = None

        print(f"Found {len(chat_models)} accessible Gemini models:")
        for model in chat_models:
            print(f" • {model.name}")
            
            # Check for Gemini 3 specific string
            if "gemini-3" in model.name.lower():
                gemini_3_models.append(model)
        
        print("-------------------------------------------")

        # Determine which model to test
        target_model_name = ""
        
        if gemini_3_models:
            print(f"✅ Success! Found {len(gemini_3_models)} model(s) explicitly matching 'gemini-3':")
            for m in gemini_3_models:
                print(f"   - {m.name}")
            
            print("\n-------- Testing ALL Found Gemini 3 Models --------")
            for model in gemini_3_models:
                target_name = model.name
                print(f"\n🧪 Testing generation with model: {target_name}...")
                try:
                    response = client.models.generate_content(
                        model=target_name,
                        contents="Hello! Please confirm you are operational."
                    )
                    
                    if response and response.text:
                        print("✅ Generation Successful!")
                        print(f"🤖 Response: {response.text.strip()}")
                    else:
                        print("⚠️  Generation completed but returned no text content.")
                        
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "quota" in error_str.lower():
                        print("⚠️  Quota Exceeded (429): You have access to this model, but your current plan/quota prevents generation.")
                    elif "404" in error_str:
                        print(f"❌ Model not found (404).")
                    elif "403" in error_str:
                        print(f"❌ Permission Denied (403).")
                    else:
                        print(f"❌ Generation failed with unexpected error: {error_str}")

        else:
            print("ℹ️  'Gemini 3' was NOT explicitly found in the model list.")
            # ... (keep existing fallback logic if desired, or just exit)
            return

    except Exception as e:
        print(f"❌ Error during API model list check: {e}")

if __name__ == "__main__":
    # Suppress warnings about future deprecation to keep output clean
    import warnings
    warnings.simplefilter("ignore")
    check_gemini_access()
