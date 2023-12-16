from pathlib import Path
import desc2cat as dc
from dataclasses import dataclass
import tyro


@dataclass
class Args:
    result_dir: Path
    '''Directory with trained results.'''


if __name__ == "__main__":
    args = tyro.cli(Args)
    model = dc.CategoryModel(args.result_dir / Path('categories.txt'),
                             args.result_dir / Path('cat_model'))
    description = "Comcast"
    print("Predicted Category:", model.predict_category(description))
