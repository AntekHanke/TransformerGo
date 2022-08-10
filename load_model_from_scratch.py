from transformers import AutoModelForSeq2SeqLM, AutoModel

path = "/home/tomek/Research/subgoal_chess_data/policy/micro/"

model = AutoModel.from_pretrained(path)


