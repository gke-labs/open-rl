#!/usr/bin/env python3
import argparse
import requests
import sys
import os
import json
import logging
# Suppress transformers logging ("PyTorch was not found..." etc.)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from datetime import datetime
from transformers import AutoTokenizer, logging as tr_logging
tr_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.ERROR)

def get_server_url():
    return os.environ.get("OPEN_RL_BASE_URL", "http://localhost:8000")

def list_adapters(args):
    url = f"{get_server_url()}/api/v1/list_adapters"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        adapters = data.get("adapters", [])
        
        print(f"{'ID':<40} | {'ALIAS':<30} | {'CREATED':<25}")
        print("-" * 100)
        
        for adapter in adapters:
            model_id = adapter.get("model_id", "N/A")
            alias = adapter.get("alias") or "N/A"
            created_at = adapter.get("created_at", "N/A")
            
            # Format timestamp if possible
            try:
                # If it's a float timestamp
                if isinstance(created_at, (int, float)):
                    dt = datetime.fromtimestamp(created_at)
                    created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
                # If it's an ISO string (from metadata.json)
                elif isinstance(created_at, str) and "T" in created_at:
                    dt = datetime.fromisoformat(created_at)
                    created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
                
            print(f"{model_id:<40} | {alias:<30} | {created_at:<25}")
            
    except Exception as e:
        print(f"Error listing adapters: {e}")
        sys.exit(1)

def chat(args):
    model_id = args.model
    if not model_id:
        print("Error: --model <model_id> is required for chat.")
        sys.exit(1)
        
    print(f"Initializing chat with model: {model_id}...")
    
    # 1. Get Base Model Info (to load correct tokenizer)
    base_url = get_server_url()
    try:
        # We use get_info with the model_id to see if server knows about it, 
        # but really we just need the base model name.
        info_resp = requests.post(f"{base_url}/api/v1/get_info", json={"model_id": model_id}, timeout=5)
        if info_resp.status_code == 200:
            info = info_resp.json()
            base_model = info.get("model_name")
            if not base_model:
                print("Error: Server has no base model loaded for this adapter. Cannot determine tokenizer.")
                sys.exit(1)
        else:
            print("Error: Could not fetch server info to determine tokenizer.")
            sys.exit(1)
            
        print(f"Using base model tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    except Exception as e:
        print(f"Error initializing: {e}")
        sys.exit(1)
        
    system_prompt = args.system_prompt or "You are a helpful assistant."
    history = [{"role": "system", "content": system_prompt}]
    
    print("\n" + "="*50)
    print(f"Chat Session Started ({model_id})")
    print(f"System Prompt: {system_prompt}")
    print("Type 'exit' or 'quit' to end.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue
                
            # Update history (temporarily, we don't actually keep history in this simple MVP server 
            # unless we implement a proper chat loop, but for 'asample' we usually just send the full context)
            # The server `asample` is stateless, so we must send full history if we want context.
            # However, `asample` expects `prompt_token_ids`.
            
            history.append({"role": "user", "content": user_input})
            
            # Tokenize
            text = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Send to server
            payload = {
                "model_id": model_id,
                "prompt": {"chunks": [{"tokens": tokens}]},
                "sampling_params": {
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature
                },
                "num_samples": 1
            }
            
            resp = requests.post(f"{base_url}/api/v1/asample", json=payload, timeout=30)
            resp.raise_for_status()
            
            # Retrieve future (polling)
            req_id = resp.json().get("request_id")
            if not req_id:
                print("Error: No request_id returned.")
                continue
                
            # Poll
            import time
            while True:
                poll_resp = requests.post(f"{base_url}/api/v1/retrieve_future", json={"request_id": req_id}, timeout=5)
                poll_data = poll_resp.json()
                
                if poll_data.get("type") == "try_again":
                    time.sleep(0.1)
                    continue
                elif poll_data.get("type") == "RequestFailedResponse":
                    print(f"Error: {poll_data.get('error_message')}")
                    break
                elif poll_data.get("type") == "sample":
                    # success
                    sequences = poll_data.get("sequences", [])
                    if sequences:
                        output_tokens = sequences[0].get("tokens", [])
                        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                        print(f"Assistant: {output_text}\n")
                        history.append({"role": "assistant", "content": output_text})
                    break
                else:
                    print(f"Unknown response type: {poll_data.get('type')}")
                    break
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during chat: {e}")

def main():
    parser = argparse.ArgumentParser(description="Open-RL CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available adapters")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with an adapter")
    chat_parser.add_argument("--model", required=True, help="Model ID (adapter ID) to use")
    chat_parser.add_argument("--system-prompt", help="System prompt to use")
    chat_parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_adapters(args)
    elif args.command == "chat":
        chat(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
