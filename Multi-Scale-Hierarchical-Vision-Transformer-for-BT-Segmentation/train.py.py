import argparse
import os
import torch
from torch.utils.data import DataLoader
from models import MS_HVT
from data import BraTSDataset
from utils.losses import CombinedLoss
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = BraTSDataset(args.data_dir, split='train')
    val_dataset = BraTSDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    model = MS_HVT().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, segs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, segs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, segs in val_loader:
                imgs, segs = imgs.to(device), segs.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, segs)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f'Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

if __name__ == '__main__':
    main()