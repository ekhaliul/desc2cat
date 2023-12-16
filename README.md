Sharing this for anyone struggling with getting automatic categories for credit card or bank transactions.

### Setup:

```bash
python3 -m venv venv; source venv/bin/activate; pip install -e .
```

### Data preparation

You will need to create a folder  data and two files:\
data/\
├── transactions.csv\
└── categories_map.csv

transactions.csv is the file that you download from mint with transactions listed.
You should have columns similar to this: 
```
"Date","Description","Original Description","Amount","Transaction Type","Category","Account Name","Labels","Notes"
```

categories_map.csv contains mapping for Mint categories and categories that you want to keep track of. For example you can have something like this:

```csv
Sell,Investment
Service & Parts,Automotive
Service Fee,Professional Services
Shipping,Shopping
Shopping,Shopping
Sporting Goods,Shopping
```
### Training

```shell
python scripts/train.py --input_dir data --output output --valid_accounts "CREDIT CARD" "PREMIER PLUS CKG"
```
This may take a while.

### Running
```shell
python scripts/inferrence.py --result-dir output
```