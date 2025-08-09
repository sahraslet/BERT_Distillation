'''Tokenizer wrapper for BERT Distillation'''
from transformers import AutoTokenizer
from typing import Optional, Union, List
from pathlib import Path

class Tokenizer:
    def __init__(self, tokenizer_path: Union[str, Path], use_fast: bool = True):
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)

    def tokenize(self, text: Union[str, List[str]],
                 max_length: Optional[int] = None,
                 padding: str = 'max_length',
                 truncation: bool=True,
                 return_tensors: str = 'pt' ) -> dict:
        '''Tokenizes the input text.'''
        return self.tokenizer(text,
                              max_length=max_length,
                              truncation=truncation,
                              padding=padding,
                              return_tensors=return_tensors
                              )

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decodes token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Saves the tokenizer to the specified directory."""
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.tokenizer.vocab_size

    def mask_token(self) -> str:
        """Returns the ID of the [MASK] token."""
        return self.tokenizer.mask_token_id