import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ZenDataset(Dataset):
    def __init__(self, text_file, model_id="allenai/Olmo-3-7B-Base", block_size=2048):
        # Split on end of text, but keep the token
        # Not sure if this is how the olmo3 tokenizer even represents end of tokens
        print("Loading tokenizer, hope you chose olmo...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.block_size = block_size
        with open(text_file, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        examples = self.raw_text.split("<|end_of_text|>")
        print(f"Found {len(examples)} examples. Time to pack dem biatches")

        self.input_ids = []
        for ex in examples:
            if not ex.strip():
                continue
            cleaned_ex = ex.replace("<|start_of_text|>", "")
            tokenized = self.tokenizer.encode(cleaned_ex) + [
                self.tokenizer.eos_token_id
            ]
            self.input_ids.extend(tokenized)
        print(
            f"You got {len(self.input_ids)} examples and {len(self.input_ids) // self.block_size} batches"
        )

    def __len__(self):
        return len(self.input_ids) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 for label
        chunk = torch.tensor(self.input_ids[start:end])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# Quick Test Block (Runs only if you execute this file directly)
if __name__ == "__main__":
    ds = ZenDataset("zen_training_data.txt")
    print(f"\n✅ Dataset loaded. First item shape: {ds[0][0].shape}")
    print("First few tokens decoded:")
    print(ds.tokenizer.decode(ds[0][0][:50]))
