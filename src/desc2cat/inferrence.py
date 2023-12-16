from pathlib import Path

import torch
from transformers import BertTokenizer

from .dataset import CategoriesHandler
from .model import BertModel


class CategoryModel:

    def __init__(self, categories_path: Path, ckpt_path: Path):
        self.categories = CategoriesHandler.read(categories_path)
        # Load the model
        model = BertModel(len(self.categories))  # Initialize your model
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('mps')))
        model.eval()  # Set the model to evaluation mode
        self.model = model

    @staticmethod
    def prepare_input(description, max_length=100):
        """
        Tokenize and prepare the input text.
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoding = tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def predict_category(self, description):
        """
        Make a prediction for the given description.
        """
        input_ids, attention_mask = self.prepare_input(description)
        with torch.no_grad():  # No need to compute gradients for inference
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Assuming the output is logits; modify if your model's output is different
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return self.categories[predicted_class]
