import torch

torch.cuda.empty_cache()

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, TrainingArguments, Trainer

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
        item = {'input_ids' : [0,1,0,1], 'attention_maks': [1,1,1,1], 'labels': [2,2,2,2]}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return 1000
#
#

train_dataset = ChesHFDataset()
val_dataset = ChesHFDataset()
test_dataset = ChesHFDataset()

model = BartForConditionalGeneration(BartConfig(vocab_size=5, max_position_embeddings=16, encoder_ffn_dim=512, decoder_ffn_dim=512))
# model = BartForConditionalGeneration(BartConfig(vocab_size=5, max_position_embeddings=16, encoder_layers=4, decoder_layers=4, encoder_ffn_dim=128, decoder_ffn_dim=128))

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# encodings = tokenizer(train_texts, truncation=True, padding=True)

# input = tokenizer("Ufo ufo wie lepiej", return_tensors="pt").input_ids

# uu = torch.IntTensor([[10,20,30]])

# outputs = model.generate(torch.IntTensor([[10,20,30]]), num_beams=4, num_return_sequences=2)



# print(outputs)