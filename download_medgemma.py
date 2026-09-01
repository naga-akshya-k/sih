"""
Script to download official Google MedGemma 1.5 4B IT weights to local Hugging Face cache.
"""
import sys
from huggingface_hub import snapshot_download, get_token

MODEL_ID = "google/medgemma-1.5-4b-it"

def main():
    token = get_token()
    print("=" * 60)
    print(f"COLONPATH-AI: Downloading {MODEL_ID}")
    print("=" * 60)
    
    if not token:
        print("[!] No Hugging Face token found.")
        print("Please run: huggingface-cli login")
        print("Or set the HF_TOKEN environment variable.")
        sys.exit(1)
        
    print(f"[*] Found Hugging Face token (length: {len(token)}).")
    print(f"[*] Connecting to https://huggingface.co/{MODEL_ID}...")
    
    try:
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            token=token,
            resume_download=True
        )
        print("\n[OK] DOWNLOAD COMPLETE!")
        print(f"[OK] Model weights cached at: {local_path}")
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            print("\n" + "!" * 60)
            print("GOOGLE ACCESS PERMISSION REQUIRED")
            print("!" * 60)
            print("Google MedGemma is a gated research model.")
            print("Your Hugging Face account just needs one-click approval:")
            print(f"\n1. Open this link in your browser:")
            print(f"   -> https://huggingface.co/{MODEL_ID}")
            print("\n2. Log in with your Hugging Face account.")
            print("3. Click 'Acknowledge license' / 'Agree and access repository'.")
            print("4. Re-run this script: python download_medgemma.py")
            print("!" * 60)
        else:
            print(f"\n[!] Download error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
