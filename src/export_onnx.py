import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Wrapper(torch.nn.Module):
    """
    Wrap GPT2 model so that ONNX export does not try to trace cache logic.
    Forces use_cache=False to avoid .get_seq_length() calls.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False  # critical fix for ONNX export
        )
        return outputs.logits


def main():
    print("Loading model/tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Dummy input (batch_size=1, sequence_length=8)
    text = "Hello, my name is Emran"
    inputs = tokenizer(text, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Wrap model to disable cache
    wrapped_model = GPT2Wrapper(model)

    print("Exporting to ONNX (classic exporter)...")
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        "gpt2.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17
    )

    print("✅ Export complete! Model saved to gpt2.onnx")


if __name__ == "__main__":
    main()
