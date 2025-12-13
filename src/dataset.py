import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ZenDataset(Dataset):
    def __init__(self, text_file, model_id="allenai/Olmo-3-7B-Base", block_size=2048):
        print("Loading tokenizer, hope you chose olmo...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token
        print(f"EOS token: {eos}, BOS token: {bos}")

        self.block_size = block_size
        with open(text_file, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

        # Split on actual EOS token
        examples = self.raw_text.split(eos) if eos else [self.raw_text]
        print(f"Found {len(examples)} examples. Time to pack dem biatches")

        self.input_ids = []
        for ex in examples:
            if not ex.strip():
                continue
            # Strip BOS if present (tokenizer may add it automatically)
            cleaned_ex = ex.replace(bos, "") if bos else ex
            tokenized = self.tokenizer.encode(cleaned_ex) + [
                self.tokenizer.eos_token_id
            ]
            self.input_ids.extend(tokenized)

        print(
            f"Total tokens: {len(self.input_ids)}, batches: {len(self.input_ids) // self.block_size}"
        )

    def __len__(self):
        return len(self.input_ids) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = torch.tensor(self.input_ids[start:end])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


if __name__ == "__main__":
    ds = ZenDataset("zen_training_data.txt")
    print(f"\n✅ Dataset loaded. First item shape: {ds[0][0].shape}")
    print("First few tokens decoded:")
    print(ds.tokenizer.decode(ds[0][0][:50]))
