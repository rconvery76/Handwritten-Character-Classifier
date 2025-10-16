from .cnn import *
from .data import *
from typing import List, Sequence
from torch.utils.data import Dataset
from pathlib import Path



class PTShardDataset(Dataset):
    def __init__(self, shard_paths: Sequence[Path]):
        self.items: List[tuple] = []
        for p in shard_paths:
            chunk = torch.load(p, map_location='cpu', weights_only=False)
            if not isinstance(chunk, list):
                raise ValueError(f"Expected list in shard {p}, got {type(chunk)}")
            self.items.extend(chunk)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x_np, y = self.items[idx]
        x = torch.as_tensor(x_np, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Add channel dimension for grayscale images
        elif x.ndim == 3 and x.shape[0] != 1:
            x = x.permute(2, 0, 1).contiguous()  # Change HWC to CHW

        y = int(y)

        return x, y
    
def create_dataloader(proc_dir: Path, batch_train=128, batch_eval=256, val_frac=0.1, seed=42):
    train_dir = proc_dir / 'train'
    test_dir = proc_dir / 'test'

    train_shards = sorted(train_dir.glob('shard_*.pt'))
    test_shards = sorted(test_dir.glob('shard_*.pt'))

    full_train = PTShardDataset(train_shards)
    test = PTShardDataset(test_shards)

    if not train_shards or not test_shards:
        raise FileNotFoundError(f"No shards found in {train_dir} or {test_dir}")

    # Split training shards into training and validation sets
    num_val_shards = int(len(full_train) * val_frac)
    num_train_shards = len(full_train) - num_val_shards

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [num_train_shards, num_val_shards], generator=g
    )

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_eval, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage
    train_loader, val_loader, test_loader = create_dataloader(OUTPUT_DIR)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    for x, y in train_loader:
        print(f"Batch x shape: {x.shape}, y shape: {y.shape}")
        break

