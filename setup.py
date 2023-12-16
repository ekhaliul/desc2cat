from setuptools import setup, find_packages

setup(
    name='desc2cat',  # Use your package name here
    version='0.1.0',
    description='Train and run a model to convert transactions to categories',
    author='Eldar Khaliullin',
    author_email='desc2cat@drively.co',
    url='https://github.com/ekhaliul/desc2cat',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'torch',
        'transformers',
        'numpy',
        'scikit-learn',
        'icecream',
        'pandas',
        'dataclasses',
        'tyro',
        'pathlib'
    ]
)
