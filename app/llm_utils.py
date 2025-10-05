from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL_ID = "microsoft/DialoGPT-medium"
LLM_MODEL_PATH = "./models/DialoGPT-medium"

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_safetensors=False, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, use_safetensors=False, trust_remote_code=False)
    return tokenizer, model