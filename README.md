## Evaluation on internal lab dataset
As part of my work for the [Chestek Lab](https://chestekresearch.engin.umich.edu/), I used the POYO framework to assess whether including past days in the training set improves test performance of future days. Specifically, this was in regard to an internal lab 174 day dataset that contained neural channel spike times with the corresponding finger kinematics. To do this, I trained an 11.8M parameter model on varying amount of training days to decode 2D finger velocities. The training sets consisted of 174 days, 100 days, 50 days, 20 days, and then each of the last 20 days individually. This ensured that the train and test splits of the last 20 days of the dataset were present in each model, giving a consistent test set to evaluate the models. Of those last 20 days, 4 had single day models with an R2 score <= 0.05, likely the result of poor neural recordings on these days. The results can be seen in the two box plots below, one with all 20 days and one with the 4 "bad" days removed. 

![alt text](https://github.com/omerbe/torch_brain/blob/main/poyo_scaling_combined_day_plot.png)

As can be seen above, adding more days to the training dataset does indeed improve performance. In regards to the entire 20 day test set, the 174 day model had a mean and median that were both 10% greater than that of the aggregated  single day models.

The remainder of this ReadMe is that of the original torch_brain repo.

<p align="left">
    <img height="250" src="https://torch-brain.readthedocs.io/en/latest/_static/torch_brain_logo.png" />
</p>

[Documentation](https://torch-brain.readthedocs.io/en/latest/) | [Join our Discord community](https://discord.gg/kQNKA6B8ZC)

[![PyPI version](https://badge.fury.io/py/pytorch_brain.svg)](https://badge.fury.io/py/pytorch_brain)
[![Documentation Status](https://readthedocs.org/projects/torch-brain/badge/?version=latest)](https://torch-brain.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml)
[![Discord](https://img.shields.io/discord/1338561153089146962?label=Discord&logo=discord)](https://discord.gg/kQNKA6B8ZC)

**torch_brain** is a Python library for various deep learning models designed for neuroscience.

### Features
+ Multi-recording training
+ Optimized data loading with with on-demand data access -- only loads data when needed
+ Advanced samplers that enable arbitrary slicing of data on the fly
+ Advanced data collation strategies including chaining and padding
+ Support for arbitrary neural and behavioral modalities
+ Collection of useful nn.Modules like stitchers, multi-output readouts, infinite vocab embeddings, etc.
+ Collection of neural and behavioral transforms and augmentation strategies
+ Implementations of various deep learning models for neuroscience

### List of implemented models

+ [POYO: A Unified, Scalable Framework for Neural Population Decoding (Azabou et al. 2023)](examples/poyo)
+ More coming soon...


## Installation
torch_brain is available for Python 3.9 to Python 3.11

To install the package, run the following command:
```bash
pip install pytorch_brain
```

## Contributing
If you are planning to contribute to the package, you can install the package in
development mode by running the following command:
```bash
pip install -e ".[dev]"
```

Install pre-commit hooks:
```bash
pre-commit install
```

Unit tests are located under test/. Run the entire test suite with
```bash
pytest
```
or test individual files via, e.g., `pytest test/test_binning.py`


## Cite

Please cite [our paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html) if you use this code in your own work:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
