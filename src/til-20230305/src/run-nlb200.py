import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", cache_dir="data/external"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M", cache_dir="data/external"
)
if torch.cuda.is_available():
    # model = model.to("mps")
    model = model.to("cpu")


article = """
I watched a lot of interesting animation last week.
I heard it was nice today, but it didn't rain!
I saw a lot of interesting animation last week, but this week I couldn't watch any of it. Hopefully I'll be able to watch it next week...
"""
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["jpn_Jpan"], max_length=100
)
result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(result)
