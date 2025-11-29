import torch
import argparse
import yaml
from tokenizer.tokenizer import Tokenizer
from LM_basics.full_transformer import Transformer_LM


class TextGenerator:
    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate(self, prompt: str, max_len = 200, temperature=0.8, top_p = 0.9, repetition_penalty=1.2) -> str:
        input_ids: list[int] = self.tokenizer.encode(prompt)
        generated: list[int] = input_ids.copy()

        with torch.no_grad():
            for step in range(max_len):
                inputs = torch.tensor([input_ids], device=self.device)
                # take the probability distribution of the last token
                logits = self.model(inputs)[0, -1, :] # (vocab_length)
                # repetition_penalty
                for token_id in set(generated):
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty
                logits /= temperature # the higher the temperature is, the more random the distribution will be
                probs = torch.softmax(logits, dim=-1)
                # sort
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # find the cumulative probs that is greater than top_p
                cutoff = cumulative_probs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                # block tokens that are not in nucleus
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # sample according to probs
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
                next_token_id = int(next_token)
                if self.tokenizer.decode([next_token_id]) == "<|endoftext|>":
                    print("\n(The End)")
                    break

                generated.append(int(next_token_id))
                input_ids.append(int(next_token_id))
                print(self.tokenizer.decode([next_token_id]), end="")

                if step == max_len - 1:
                    print(f"\n(The rest content is truncated. step: {step})")

        return self.tokenizer.decode(generated)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script with config support")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    parser.add_argument("--prompt", type=str, help="prompt")
    parser.add_argument("--max_len", type=int, help="maximum tokens that the model can generate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--temperature", type=float, help="temperature")
    parser.add_argument("--top_p", type=float, help="top_p sample")
    parser.add_argument("--repetition_penalty", type=float, help="repetition penalty")
    return parser.parse_args()


def load_config(args):
    # load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # command-line arguments cover in prior
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def main():
    args = parse_args()
    cfg = load_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer_LM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["RoPE_theta"]
    ).to(device)
    checkpoint = torch.load("checkpoints/final_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")

    tokenizer = Tokenizer.from_files(
        vocab_filepath="tokenizer/vocab.pkl",
        merges_filepath="tokenizer/merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    while True:
        prompt: str = input("Prompt: ")
        text = generator.generate(
            prompt=prompt,
            max_len=cfg["max_len"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"]
        )
        print("\n\nTotal tokens: ", len(text))


if __name__ == "__main__":
    main()