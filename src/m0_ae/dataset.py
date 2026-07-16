import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import h5py
import bisect

preprocess = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
])

def normalize_image(x):
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)

def standardize_image(x):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True)
    return (x - mean) / (std + 1e-8)

class MultiH5Dataset(Dataset):
    def __init__(self, h5_files, dataset_name="patches"):
        self.h5_files = h5_files
        self.dataset_name = dataset_name

        self.lengths = []
        for path in h5_files:
            with h5py.File(path, "r") as f:
                self.lengths.append(len(f[dataset_name]))

        # Prefix sums for global indexing
        self.cumulative = [0]
        for l in self.lengths:
            self.cumulative.append(self.cumulative[-1] + l)

        self._files = [
            h5py.File(path, "r") for path in self.h5_files
        ]

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[file_idx]

        x = self._files[file_idx][self.dataset_name][local_idx]

        x = torch.from_numpy(x).float()/255

        x = preprocess(x)
        # x = standardize_image(x)

        return x

    def __del__(self):
        if self._files is not None:
            for f in self._files:
                try:
                    f.close()
                except Exception:
                    pass 