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

        train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        test_loader = DataLoader(test, batch_size=batch_eval, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

        return train_loader, val_loader, test_loader
    
def run_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    train = optimizer is not None
    model.train(train)
    total, correct, loss = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss / total, correct / total

def evaluate_confusion(model, loader, num_classes, device='cpu'):

    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion[int(t), int(p)] += 1

    per_class_acc = confusion.diagonal() / np.clip(confusion.sum(axis=1), 1, a_max=None)

    return confusion, per_class_acc

    
    

if __name__ == '__main__':
    # Example usage
    train_loader, val_loader, test_loader = PTShardDataset.create_dataloader(OUTPUT_DIR, batch_train=128, batch_eval=256)

    model = SimpleCNN(CNNConfig(input_channels=1, num_classes=26, conv_layers=(32,64,128), dropout=0.3)).to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    epochs = 15
    best_val = 0.0
    save_path = Path('best_model.pt')
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device='cpu')
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device='cpu')

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)  # uses validation accuracy
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr != prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {curr_lr:.2e}")

        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved with Val Acc={best_val:.4f}')

