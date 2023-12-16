from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from transformers import BertTokenizer
import numpy as np
from pathlib import Path
from .utils import CategoriesHandler
from typing import List
from icecream import ic


class Validator:
    def __init__(self, tokenizer, valid_accounts):
        self.tokenizer = tokenizer
        self.valid_accounts = valid_accounts

    def is_valid_description(self, description):
        """
        Checks if the description is valid. You can define your own criteria of validity.
        """
        try:
            # Example validity check: can it be tokenized without errors?
            _ = self.tokenizer.encode(description, add_special_tokens=True)
            return True
        except ValueError:
            return False

    def is_valid_account(self, account):
        return account in self.valid_accounts


class TransactionsDataset(Dataset):
    def __init__(self, path: Path, valid_accounts: List[str]):
        ic(valid_accounts)
        ic.configureOutput(prefix='input path: ')
        ic(path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataframe = pd.read_csv(path / Path('transactions.csv'))
        self.categories_map = CategoriesHandler.read_mapping(path / Path("categories_map.csv"))
        # Filter out rows with invalid descriptions
        self.validator = Validator(self.tokenizer, valid_accounts)
        self.dataframe = self.dataframe[
            self.dataframe['Original Description'].apply(self.validator.is_valid_description)]
        self.dataframe = self.dataframe[
            self.dataframe['Account Name'].apply(self.validator.is_valid_account)]
        ic.configureOutput(prefix='number of rows: ')
        ic(len(self.dataframe))
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        categories = [[self.categories_map[cat[0]]] for cat in
                      self.dataframe[['Category']].values]
        self.onehot_encoder.fit(categories)
        self.max_length = 100
        self.categories = self.onehot_encoder.categories_[0]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        description = self.dataframe.iloc[idx]['Original Description']
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        token_ids = encoding['input_ids'].squeeze(0)  # remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        category = self.categories_map[self.dataframe.iloc[idx]['Category']]
        encoded_category = self.onehot_encoder.transform([[category]]).astype(np.float32)
        return token_ids, encoded_category[0], attention_mask
