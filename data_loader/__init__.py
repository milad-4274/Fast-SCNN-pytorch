from .cityscapes import CitySegmentation
from .lip_parsing import LIPSegmentation

datasets = {
    'citys': CitySegmentation,
    'lips': LIPSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
