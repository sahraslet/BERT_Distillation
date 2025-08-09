'''Dataset class for general distillation'''
from torch.utils.data import Dataset

class DistilDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        tokenized = self.tokenizer.tokenize(text,
                                            max_length=self.max_length,
                                            padding = 'max_length',
                                            truncation = True,
                                            return_tensors="pt")
        return tokenized
