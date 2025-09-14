#!/usr/bin/env python3
"""
Ollama Setup Script for Prescription Scanner
This script helps check and set up Ollama with Gemma3 model for offline operation.
"""

import requests
import json
import subprocess
import sys
import os

OLLAMA_URL = 'http://localhost:11434'
MODEL_NAME = 'gemma2'

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def list_available_models():
    """List available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except requests.exceptions.RequestException:
        return []

def pull_model(model_name):
    """Pull a model from Ollama"""
    try:
        print(f"Pulling {model_name} model...")
        payload = {"name": model_name}
        response = requests.post(f"{OLLAMA_URL}/api/pull", json=payload, timeout=300)
        
        if response.status_code == 200:
            print(f"Successfully pulled {model_name}")
            return True
        else:
            print(f"Failed to pull {model_name}: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error pulling model: {e}")
        return False

def test_model(model_name):
    """Test if model is working"""
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello, please respond with 'Model is working'",
            "stream": False
        }
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Model test response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"Model test failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error testing model: {e}")
        return False

def main():
    print("🤖 Ollama Setup for Prescription Scanner")
    print("=" * 50)
    
    # Check if Ollama is running
    print("1. Checking if Ollama is running...")
    if check_ollama_running():
        print("✅ Ollama is running")
    else:
        print("❌ Ollama is not running")
        print("\nTo install and start Ollama:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start Ollama: ollama serve")
        print("3. Run this script again")
        return False
    
    # List available models
    print("\n2. Checking available models...")
    models = list_available_models()
    print(f"Available models: {models}")
    
    # Check if Gemma3 is available
    gemma_models = [m for m in models if 'gemma' in m.lower()]
    
    if not gemma_models:
        print(f"\n3. {MODEL_NAME} not found. Attempting to pull...")
        if pull_model(MODEL_NAME):
            print(f"✅ Successfully pulled {MODEL_NAME}")
        else:
            print(f"❌ Failed to pull {MODEL_NAME}")
            print("Alternative models you can try:")
            print("- ollama pull llama2")
            print("- ollama pull mistral")
            return False
    else:
        print(f"✅ Gemma model found: {gemma_models[0]}")
    
    # Test the model
    print(f"\n4. Testing {MODEL_NAME} model...")
    if test_model(MODEL_NAME):
        print(f"✅ {MODEL_NAME} is working correctly")
    else:
        print(f"❌ {MODEL_NAME} test failed")
        return False
    
    print("\n🎉 Setup complete! Your prescription scanner is ready for offline use.")
    print("\nTo use the application:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Start the Flask app: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)