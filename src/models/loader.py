import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cpu", dtype: str = "float32"):
    """
    Loads model and tokenizer from Hugging Face Hub.
    """
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Map dtype string to torch dtype
    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    
    print(f"Loading model {model_id} on {device} with dtype {torch_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    
    # Configure padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer
