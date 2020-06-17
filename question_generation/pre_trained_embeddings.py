import torch

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

BERT_ENCODER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
