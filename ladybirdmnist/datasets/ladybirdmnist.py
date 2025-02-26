import os
import os.path
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset

class LadybirdMNIST(VisionDataset):
    data_dict = {
        'morph-28': [
            ['morph-28', 'b88dda51fca4df2e46559e21dc9a6fdc'],
        ],
        'morph-128': [
            ['morph-128', '3b118f6b046e8bc05387e4ac18d0fe61'],
        ],
        'pattern-28': [
            ['pattern-28', '6a36372bc3844a7dbb0dc901fbd085c5'],
        ],
        'pattern-128': [
            ['pattern-128', '2b32ffa3a2c9dea03695a7fb3180b9c2'],
        ],
        'pde': [['pde', '44272373c52e65ff8ebd41681f0096bb']],
        'state': [
                    ['state_0', 'b2a64b032fb77cc68af18169371558de'],
                    ['state_1', 'df4987ea0baf52634f4e59cb50323e8a'],
                    ['state_2', '673f0262df3b33b6f5ec28321a129502'],
                    ['state_3', '2b9babc0a2a96877ee2e797e2f56f665'],
                    ['state_4', '55ee43998b418f1c6172ca81f950a861'],
                  ],
        'meta': [['meta', 'cbd5a6ab3afdd305a62ad4986035bb89']],
        # '_label': [['label', '4a74f24622c11f53abad657b124698e6'],],
    }

    download_url = {
        'morph-28': ['1soT5YfGjZLwsYeGLAEiznCP8wzqdkGYb', 'c6e09f74144c30a0e159c825be3ede35'],
        'morph-128': ['1HYbQbGBMDf78NZUAXzW2cOAikc23uooU', 'e9fa01d8b20c1f63aab41a9d3d90a2bb'], 
        'pattern-28': ['1PPX2S5rSXeP9CVCOza5le93m5xVAcdSX', 'f3f127866a28ae5668fc12b64c7b1dc4'],
        'pattern-128': ['1r9Ntv9uHPKhZ7Er-o-hLcBfqb4iQ0UC7', 'b2528406dbd292494e053d177a6b0328'],
        'pde': ['1-6HmLhYe2-53jwa3IE4zvTDMWFdG95uj', '51e7223f52b9f6099da72e85b5d98c7b'],
        'state': ['1JgStxSiBZaUrC8-Estyl_nBjRw9Qh9AN', '10f8046e0eabf186881ee75131d3cb55'],
        'meta': ['1BnsAuA0hxJjkByqBN6eynADC02YLYXf-', '2bd4b7dd4ca23aeee01f68c1ecd00c57'],
        # '_label': ['1Efglx5h1GobmBGz47plfZfNnKKDJ9TWB', '36ad544d71958ed1fdd3d8c9ec46d88f'],
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[bool, str] = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        dataset: List[str] = ['morph-28'],
        random_seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        super().__init__(root, transform=transform)

        self.dataset = dataset
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        self.data: Any = []
        self.label = []

        for dataset in self.dataset:
            all_data = []
            tmp_label = []
            data = []
            for file_name, checksum in self.data_dict[dataset]:
                file_path = os.path.join(self.root, 'raw', file_name)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f)
                all_data.extend(entry['data'])

            if split == True or split == 'train':
                for i in range(10): 
                    tmp_data = all_data[i*7000:(i+1)*7000]
                    np.random.shuffle(tmp_data)
                    data.extend(tmp_data[:6000]) # 6000 for train
                    tmp_label.extend([i] * 6000)

            elif split == False or split == 'test':
                for i in range(10):
                    tmp_data = all_data[i*7000:(i+1)*7000]
                    np.random.shuffle(tmp_data)
                    data.extend(tmp_data[6000:])
                    tmp_label.extend([i] * 1000)

            elif split == 'all':
                for i in range(10):
                    tmp_data = all_data[i*7000:(i+1)*7000]
                    np.random.shuffle(tmp_data)
                    data.extend(tmp_data)
                    tmp_label.extend([i] * 7000)

            else:
                raise ValueError(f"Invalid split: {split}")
            self.data.append(data)
            self.label.extend(tmp_label)

        # for file_name, checksum in self.data_dict['_label']:
        #     file_path = os.path.join(self.root, 'raw', file_name)
        #     with open(file_path, "rb") as f:
        #         entry = pickle.load(f)
        #     self.targets.extend(entry)

        self._load_meta()

        if shuffle:
            self._shuffle_data()

    def _shuffle_data(self) -> None:
        rng = np.random.RandomState(self.random_seed)
        indices = np.arange(len(self.data[0]))
        rng.shuffle(indices)
        
        for i in range(len(self.data)):
            self.data[i] = [self.data[i][idx] for idx in indices]
        self.label = [self.label[idx] for idx in indices]

    def _load_meta(self) -> None:
        meta_path = os.path.join(self.root, 'raw', 'meta')
        with open(meta_path, 'rb') as f:
            self.meta_dict = pickle.load(f)
        self.classes = self.meta_dict['classes']
        self.pde_params = self.meta_dict['pde_params']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        tmp_data = []
        for i, dataset in enumerate(self.dataset):
            if 'morph' in dataset or 'pattern' in dataset:
                img = self.data[i][index]
                
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)
                
                tmp_data.append(img)

            if 'pde' in dataset or 'state' in dataset:
                tmp_data.append(self.data[i][index])

        return tmp_data, self.label[index]

    def __len__(self) -> int:
        return len(self.data[0])
    
    def _check_integrity(self) -> bool:
        for dataset in self.dataset:
            for file_name, md5 in self.data_dict[dataset]:
                fpath = os.path.join(self.root, 'raw', file_name)
                if not check_integrity(fpath, md5):
                    return False
        meta_path = os.path.join(self.root, 'raw', 'meta')
        if not check_integrity(meta_path, self.data_dict['meta'][0][1]):
            return False
        return True
    
    def download(self) -> None:
        if self._check_integrity():
            return
        
        for i, dataset in enumerate(self.dataset):
            if dataset not in self.download_url.keys():
                raise ValueError(f"Invalid dataset: index {i} of {dataset}")
            
            download_and_extract_archive(
                f"https://drive.google.com/file/d/{self.download_url[dataset][0]}", 
                self.root,
                filename=f"{dataset}.tar.gz", 
                md5=self.download_url[dataset][1],
            )

            download_and_extract_archive(
                f"https://drive.google.com/file/d/{self.download_url['meta'][0]}", 
                self.root,
                filename='meta.tar.gz',
                md5=self.download_url['meta'][1],
            )