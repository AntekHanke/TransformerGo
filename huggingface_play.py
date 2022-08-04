import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model = BartForConditionalGeneration(BartConfig())
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

input = tokenizer("Ufo ufo wie lepiej", return_tensors="pt").input_ids

outputs = model.generate(input, num_beams=4, num_return_sequences=2)
print(outputs)