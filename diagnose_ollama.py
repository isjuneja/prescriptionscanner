#!/usr/bin/env python3
"""
Ollama Connection Diagnostic Tool
Helps diagnose and fix Ollama connectivity issues for the prescription scanner.
"""

import requests
import json
import subprocess
import sys
import os
import socket

OLLAMA_URL = 'http://localhost:11434'
MODEL_NAME = 'gemma3:latest'

def check_port_open(host='localhost', port=11434):
    """Check if a specific port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_ollama_process():
    """Check if Ollama process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def start_ollama():
    """Try to start Ollama service"""
    try:
        print("Attempting to start Ollama...")
        subprocess.run(['ollama', 'serve'], check=False)
        return True
    except Exception as e:
        print(f"Failed to start Ollama: {e}")
        return False

def check_ollama_api():
    """Check if Ollama API is responding"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200, response.text
    except requests.exceptions.RequestException as e:
        return False, str(e)

def check_model_available():
    """Check if the required model is available"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return MODEL_NAME in models or any('gemma' in m for m in models), models
        return False, []
    except Exception as e:
        return False, str(e)

def test_model_generation():
    """Test if model can generate responses"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": "Test prompt",
            "stream": False
        }
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Ollama Connection Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Check if Ollama is installed
    print("\n1. Checking if Ollama is installed...")
    if check_ollama_installed():
        print("✅ Ollama is installed")
    else:
        print("❌ Ollama is not installed")
        print("\n🔧 SOLUTION:")
        print("Install Ollama by running:")
        print("curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    
    # Step 2: Check if Ollama process is running
    print("\n2. Checking if Ollama process is running...")
    if check_ollama_process():
        print("✅ Ollama process is running")
    else:
        print("❌ Ollama process is not running")
        print("\n🔧 SOLUTION:")
        print("Start Ollama by running:")
        print("ollama serve")
        print("(Keep this terminal open - Ollama needs to run continuously)")
        return False
    
    # Step 3: Check if port 11434 is open
    print("\n3. Checking if Ollama port (11434) is accessible...")
    if check_port_open():
        print("✅ Port 11434 is open and accessible")
    else:
        print("❌ Port 11434 is not accessible")
        print("\n🔧 SOLUTION:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if another process is using port 11434")
        print("3. Try restarting Ollama")
        return False
    
    # Step 4: Check Ollama API
    print("\n4. Testing Ollama API connection...")
    api_ok, api_response = check_ollama_api()
    if api_ok:
        print("✅ Ollama API is responding")
    else:
        print("❌ Ollama API is not responding")
        print(f"Error: {api_response}")
        print("\n🔧 SOLUTION:")
        print("1. Restart Ollama: killall ollama && ollama serve")
        print("2. Check system resources (RAM/CPU)")
        print("3. Check Ollama logs for errors")
        return False
    
    # Step 5: Check if required model is available
    print(f"\n5. Checking if {MODEL_NAME} model is available...")
    model_ok, models = check_model_available()
    if model_ok:
        print(f"✅ {MODEL_NAME} model is available")
        if isinstance(models, list):
            print(f"Available models: {models}")
    else:
        print(f"❌ {MODEL_NAME} model is not available")
        if isinstance(models, list):
            print(f"Available models: {models}")
        print(f"\n🔧 SOLUTION:")
        print(f"Pull the required model:")
        print(f"ollama pull {MODEL_NAME}")
        print("This may take several minutes to download (~5GB)")
        return False
    
    # Step 6: Test model generation
    print(f"\n6. Testing {MODEL_NAME} model generation...")
    gen_ok, gen_response = test_model_generation()
    if gen_ok:
        print("✅ Model generation is working")
        print("🎉 All tests passed! Ollama is ready to use.")
    else:
        print("❌ Model generation failed")
        print(f"Error: {gen_response}")
        print("\n🔧 SOLUTION:")
        print("1. Try pulling the model again: ollama pull gemma2")
        print("2. Check available disk space")
        print("3. Restart Ollama service")
        return False
    
    print("\n" + "=" * 50)
    print("🚀 Your prescription scanner should now work properly!")
    print("If you're still having issues, try:")
    print("1. Restart your Flask app: python app.py")
    print("2. Upload a test prescription image")
    print("3. Check if you get proper analysis results")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Please fix the issues above and run this script again.")
    sys.exit(0 if success else 1)