import torch


def main():
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA (GPU) is not available.")
    print("You stared at zen olmo, and zen olmo stared back wait no thats nietzsche")


if __name__ == "__main__":
    main()
