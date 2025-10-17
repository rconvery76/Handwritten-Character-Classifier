from .train import *
from .data import *
import matplotlib.pyplot as plt
import string
import json

def test_model(model, test_loader, device='cpu'):
    model.eval()
    total, correct = 0, 0

    correct_by_class = {}


    with torch.no_grad():        
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += x.size(0)

            for t, p in zip(y.view(-1), pred.view(-1)):
                t = int(t)
                if t not in correct_by_class:
                    correct_by_class[t] = {'correct': 0, 'total': 0}
                if t == int(p):
                    correct_by_class[t]['correct'] += 1
                correct_by_class[t]['total'] += 1

    overall_acc = correct / total
    print(f'Test Accuracy: {overall_acc * 100:.2f}%')
    for cls, stats in correct_by_class.items():
        class_acc = stats['correct'] / stats['total']
        print(f'Class {cls}: Accuracy: {class_acc * 100:.2f}%')
    
    return overall_acc, correct_by_class


if __name__ == '__main__':
    # Example usage
    _, _, test_loader = PTShardDataset.create_dataloader(OUTPUT_DIR, batch_train=128, batch_eval=256)

    # Load a pre-trained model  best_model.pt
    model = SimpleCNN(CNNConfig(input_channels=1, num_classes=26, conv_layers=(32,64,128), dropout=0.3)).to('cpu')
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    overall_acc, correct_by_class = test_model(model, test_loader, device='cpu')

    # Display the results as charts
    num_classes = 26
    letters = list(string.ascii_uppercase)
    
    acc = []
    for i in range(num_classes):
        if i in correct_by_class:
            stats = correct_by_class[i]
            class_acc = stats['correct'] / stats['total']
        else:
            class_acc = 0.0
        acc.append(class_acc * 100)


    plt.bar(range(num_classes), acc)
    plt.xticks(range(num_classes), letters)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
   

    # Compute mean and std from the training dataset
    train_dataset.transform = make_tf(None, None)
    mean, std = compute_mean_std(train_dataset)
    print(f'Computed Mean: {mean}, Std: {std}')
    # Save the mean and std to a JSON file
    stats = {'mean': mean, 'std': std}
    with open(OUTPUT_DIR / 'mean_std.json', 'w') as f:
        json.dump(stats, f)
        
    #display the means and stds as bar charts
    plt.figure(figsize=(6,4))
    plt.bar(['Mean', 'Std'], [mean, std])
    plt.title('Dataset Mean and Std')
    plt.ylabel('Value')
    plt.show()
