import desc2cat as dc
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import tyro
from pathlib import Path
from typing import List


@dataclass
class Args:
    input_dir: str
    '''Input directory containing transactions.csv and categories_map.csv file which will be used for training'''
    output: str
    '''Directory where checkpoint and other files will be saved'''
    valid_accounts: List[str]
    '''List of all accounts to use for training, for example "CREDIT CARD". Separate values by space.'''
    device: str = 'mps'
    '''Device to use for training. cuda, mps, cpu'''


if __name__ == '__main__':
    args = tyro.cli(Args)
    out_path = Path(args.output)
    out_path.mkdir(exist_ok=True)
    full_dataset = dc.TransactionsDataset(Path(args.input_dir), args.valid_accounts)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = dc.BertModel(len(full_dataset.categories))
    dc.train_model(model, train_loader, val_loader, device=args.device)
    model.to('cpu')
    torch.save(model.state_dict(), out_path / Path('cat_model'))
    dc.CategoriesHandler.write(Path(args.output) / Path('categories.txt'), full_dataset.categories)
