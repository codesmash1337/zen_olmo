import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
# We use the 7B version to fit on consumer GPUs (RTX 3090/4090)
MODEL_ID = "allenai/OLMo-2-1124-7B"


def print_gpu_utilization():
    if torch.cuda.is_available():
        print(
            f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        print(
            f"   GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )


def main():
    print(f"🚀 Verifying environment for {MODEL_ID}...")

    # 1. CHECK CUDA
    if not torch.cuda.is_available():
        print("❌ CRITICAL: CUDA not detected. Are you on a CPU instance?")
        return
    print(f"   CUDA available: {torch.cuda.get_device_name(0)}")

    # 2. DEFINE 4-BIT CONFIG (The "Magic" for 24GB VRAM)
    print("\n📦 Configuring 4-bit Quantization (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 3. LOAD MODEL
    print("⬇️  Loading Model (this may take a minute)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,  # OLMo often requires this
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Tip: Make sure you installed requirements.txt!")
        return

    # 4. RUN INFERENCE
    print("\n🧠 Running Test Inference...")
    message = ["Language modeling is"]
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        response = model.generate(
            **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
        )

    output_text = tokenizer.decode(response[0], skip_special_tokens=True)

    print("-" * 40)
    print(f"OUTPUT: {output_text}")
    print("-" * 40)

    print("\n📊 Final Memory Footprint:")
    print_gpu_utilization()
    print("\n✅ Setup Complete! You are ready to build the training loop.")


if __name__ == "__main__":
    main()
