import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import ZenDataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb
# Load the model in 4-bit.

# Attach LoRA adapters.

# Feed data batch-by-batch.

# Calculate loss and update weights.


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError(
        "CUDA is not available. Please use a machine with a CUDA-enabled GPU."
    )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Base",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.01,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

zen_dataset = ZenDataset("data/zen_training_data.txt")
total_size = len(zen_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size
train, val = random_split(zen_dataset, [train_size, val_size])

print(f"Train size {len(train)}")
batch_size = 4
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_loader) * 10
)

# W&B logging
wandb.init(
    project="zen-finetune",
    config={
        "epochs": 10,
        "batch_size": batch_size,
        "lr": 2e-5,
    },
)
epochs = 10
validation_steps = 100


def run_validation(tokenizer, sample_prompt="If you see "):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(input_ids=x, labels=y)
            total_loss += output.loss.item()
    avg_loss = total_loss / len(val_loader)

    # Sample output
    input_ids = tokenizer.encode(sample_prompt, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=100,
        pad_token_id=tokenizer.eos_token_id,
    )
    sample_output = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"Validation loss: {avg_loss}")
    print(f"Sample output:\n{sample_output}\n")

    wandb.log(
        {
            "val_loss": avg_loss,
            "sample_output": wandb.Html(
                f"<pre>{sample_output}</pre>"
            ),  # Optional: log to W&B
        }
    )

    model.train()
    return avg_loss


model.train()
for epoch in range(epochs):
    total_loss = 0
    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(input_ids=x, labels=y)
        loss = output.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += output.loss.item()
        # Logging
        wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if (index + 1) % validation_steps == 0:
            run_validation(zen_dataset.tokenizer)

    print(f"Epoch {epoch + 1}/{epochs} complete")
    if (epoch + 1) % 2 == 0:
        model.save_pretrained(f"zen_lora_adapter_epoch{epoch + 1}")

# Save the adapter weights
model.save_pretrained("zen_lora_adapter")
wandb.finish()
