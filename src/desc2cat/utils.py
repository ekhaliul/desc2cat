import csv
from pathlib import Path


class CategoriesHandler:

    @classmethod
    def write(cls, path, categories):
        with open(path, 'w') as file:
            for item in categories:
                file.write("%s\n" % item)

    @classmethod
    def read(cls, path):
        # Reading the file and storing the lines in a list
        with open(path, 'r') as file:
            res = file.readlines()

        # Stripping the newline character from each string
        res = [line.strip() for line in res]
        return res

    @classmethod
    def read_mapping(cls, path: Path) -> dict:
        result = {}
        with Path.open(path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    key, value = row[0], row[1]
                    result[key] = value
        return result
