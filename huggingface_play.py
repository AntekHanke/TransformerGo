import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from datasets import load_dataset

# dataset = load_dataset("yelp_review_full")
#
class ChesHFDataset(torch.utils.data.Dataset):
    # def __init__(self, encodings, labels):
    #     self.encodings = encodings
    #     self.labels = labels

    def __init__(self):
        pass

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {'input_ids' : [0,1,0,1], 'attention_maks': [1,1,1,1], 'labels': [1,0,1,0]}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return 100
#
#
# model = BartForConditionalGeneration(BartConfig())

train_texts = ["elo", "melo", "Morelo"]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

encodings = tokenizer(train_texts, truncation=True, padding=True)

input = tokenizer("Ufo ufo wie lepiej", return_tensors="pt").input_ids

uu = torch.IntTensor([[10,20,30]])

# outputs = model.generate(torch.IntTensor([[10,20,30]]), num_beams=4, num_return_sequences=2)



# print(outputs)