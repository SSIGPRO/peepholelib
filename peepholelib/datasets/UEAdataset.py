from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from peepholelib.datasets.datasetWrap import DatasetWrap

# UEA Class Registery
class UEAClassRegistry:
    _cache = {}
    @staticmethod
    def build(dataset_name, raw_labels):
        unique_labels = sorted(set(raw_labels))
        mapping = {
            label: idx
            for idx, label in enumerate(unique_labels)
        }
        UEAClassRegistry._cache[dataset_name] = mapping
        return mapping
    @staticmethod
    def get(dataset_name):
        return UEAClassRegistry._cache.get(dataset_name)
    
# Torch Dataset
class TSDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        return {
            "timeseries": x,
            "label": self.y[idx]
        }
# UEA data wrapper
class TSDataWrap(DatasetWrap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = kwargs.get("path")
        if path is None:
            raise ValueError(
                "UEA dataset path must be provided."
            )
        self.root = Path(path)
        self.dataset_name = self.root.name

        # transforms
        self.train_ratio = kwargs.get("train_ratio",0.8)
        self.seed = kwargs.get("seed",42)
        
        # Variable-length handling
        self.variable_length = kwargs.get("variable_length","pad")
        valid_modes = {
            "auto",
            "pad",
            "truncate",
            "interpolate",
            "error"
        }

        if self.variable_length not in valid_modes:
            raise ValueError(
                f"Unknown variable_length mode: "
                f"{self.variable_length}"
            )
        self.__dataset__ = {}
        self.__load_data__()
        
    # Load data
    def __load_data__(self):
        train_file = (self.root / f"{self.dataset_name}_TRAIN.ts")
        test_file = (self.root /f"{self.dataset_name}_TEST.ts")

        if not train_file.exists():
            raise FileNotFoundError(
                f"TRAIN file not found:\n{train_file}"
            )

        if not test_file.exists():
            raise FileNotFoundError(
                f"TEST file not found:\n{test_file}"
            )
        # Load Raw Train
        train_samples, train_y = self._load_ts(train_file)

        # Load Raw test
        test_samples, test_y = self._load_ts(test_file)
        
        # Compute common target length for all samples
        target_length = self._compute_target_length(train_samples,test_samples)
        
        # Prepare Data
        train_X = self._prepare_samples(train_samples,target_length)
        test_X = self._prepare_samples(test_samples,target_length)
        
        # Create Datasets
        full_train_ds = TSDataset(train_X,train_y)
        n_total = len(full_train_ds)
        n_train = int(self.train_ratio * n_total)
        n_val = n_total - n_train
        train_ds, val_ds = torch.utils.data.random_split(
            full_train_ds,
            [n_train, n_val],
            generator=torch.Generator()
            .manual_seed(self.seed)
        )
        test_ds = TSDataset(test_X,test_y)
        self.__dataset__ = {
            "UEA-train": train_ds,
            "UEA-val": val_ds,
            "UEA-test": test_ds
        }

    # TS File Parser
    def _load_ts(self, file_path):
        raw_X = []
        raw_y = []
        reading_data = False
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
    
                # Skip header
                if not reading_data:
                    if line.lower() == "@data":
                        reading_data = True

                    continue

                # Parse one sample
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                raw_y.append(
                    parts[-1].strip()
                )
                sample = []
                for dim in parts[:-1]:
                    values = np.asarray(
                        [
                            float(v)
                            for v in dim.split(",")
                        ],

                        dtype=np.float32
                    )
                    sample.append(values)
                raw_X.append(sample)
        if len(raw_X) == 0:
            raise RuntimeError(
                f"No samples parsed from {file_path}"
            )
        # Label Encoding
        if "TRAIN" in file_path.name.upper():
            label_map = UEAClassRegistry.build(
                self.dataset_name,
                raw_y
            )
        else:
            label_map = UEAClassRegistry.get(
                self.dataset_name
            )
            if label_map is None:
                raise RuntimeError(
                    f"No label map for "
                    f"{self.dataset_name}"
                )
        y = np.asarray(
            [
                label_map[label]
                for label in raw_y
            ],
            dtype=np.int64
        )       
        return raw_X, y
    # Compute target length for all samples
    def _compute_target_length(self,train_samples,test_samples):
        lengths = []
        for dataset in (train_samples, test_samples):
            for sample in dataset:
                for dim in sample:
                    lengths.append(len(dim))
        min_length = min(lengths)
        max_length = max(lengths)
        print("\nSequence length statistics")
        print("--------------------------")
        print("Minimum :", min_length)
        print("Maximum :", max_length)
        if min_length == max_length:
            print("Dataset type : Equal-length")
        else:
            print("Dataset type : Variable-length")
            if self.variable_length == "error":
                raise ValueError(
                    "Variable-length dataset detected."
                )
        return max_length

    # Prepare samples
    def _prepare_samples(self,samples,target_length):
        processed = []
        
        # Check that every sample has the same number of dimensions (channels)
        expected_channels = len(samples[0])
        
        # Select processing function once
        if self.variable_length == "truncate":
            processor = self._truncate_sample
        elif self.variable_length == "interpolate":
            processor = self._interpolate_sample
        elif self.variable_length == "pad":
            processor = self._pad_sample
        elif self.variable_length == "auto":
            if self.is_variable_length:
                processor = self._pad_sample
            else:

            # Equal-length dataset:
            # No processing required.
                processor = None
        else:
            raise ValueError(
                f"Unknown variable_length mode: "
                f"{self.variable_length}"
            )
        # Process every sample
        for i, sample in enumerate(samples):
            if len(sample) != expected_channels:
                raise ValueError(
                    f"Sample {i} has {len(sample)} channels "
                    f"but expected {expected_channels}."
                )
            if processor is not None:
                sample = processor(sample,target_length)
            processed.append(
                np.stack(sample, axis=0)
            )
        return np.stack(processed,axis=0)
    # Pad Sample
    def _pad_sample(self,sample,target_length):
        padded = []
        for dim in sample:
            length = len(dim)

            # Pad shorter sequences
            if length < target_length:
                dim = np.pad(
                    dim,
                    (0, target_length - length),
                    mode="constant",
                    constant_values=0
                )
            
            # Truncate longer sequences (safety check)
            elif length > target_length:
                dim = dim[:target_length]
            padded.append(
                dim.astype(np.float32)
            )
        return padded

    # Truncate Sample
    def _truncate_sample(self,sample,target_length):
        truncated = []
        for dim in sample:
            truncated.append(
                dim[:target_length].astype(
                    np.float32
                )
            )
        return truncated
    # Interpolate Sample
    def _interpolate_sample(self,sample,target_length):
        interpolated = []
        for dim in sample:
            tensor = torch.tensor(dim,dtype=torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=target_length,
                mode="linear",
                align_corners=False
            )
            interpolated.append(
                tensor.squeeze().numpy()
            )
        return interpolated