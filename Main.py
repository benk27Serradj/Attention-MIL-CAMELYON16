from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataloader import CamelyonBags, mil_collate_fn
from Model import Attention, GatedAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CamelyonBags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--patch-size', type=int, default=700, metavar='PS',
                    help='size of each patch (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='gated_attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--data-path', type=str, default='./camelyon_data', help='Path to dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

# Load the full dataset for both training and testing
train_loader = data_utils.DataLoader(CamelyonBags(image_folder=args.data_path, 
                                                   patch_size=(args.patch_size, args.patch_size),  # Set patch size
                                                   train=True),
                                     batch_size=1,  # Increased batch size for better training
                                     shuffle=True,
                                     collate_fn=mil_collate_fn,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(CamelyonBags(image_folder=args.data_path, 
                                                  patch_size=(args.patch_size, args.patch_size),  # Set patch size
                                                  train=False),
                                    batch_size=1,  # Increased batch size for better testing
                                    shuffle=False,
                                    collate_fn=mil_collate_fn,
                                    **loader_kwargs)



print('Init Model')
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    model = GatedAttention()

# Move the model to the GPU if CUDA is available
if args.cuda:
    model.cuda()
    print("Model moved to GPU.")
else:
    print("CUDA is disabled. Model remains on CPU.")

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def train(epoch):
    print(f"training epoch: {epoch}")
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"Data shape: {data.shape}, Label : {label}")
        bag_label = label

        # Move data and label to the same device as the model
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()


        # Reset gradients
        optimizer.zero_grad()

        # Calculate loss and metrics for each patch, aggregate results
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error

        # Backward pass
        loss.backward()

        # Step
        optimizer.step()
        torch.cuda.empty_cache()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Train error: {train_error:.4f}')


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label

            # Move data and label to the same device as the model
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()

            data, bag_label = Variable(data), Variable(bag_label)

            # Calculate loss and get predicted labels for each patch
            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.item()
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            # Collect true and predicted labels for the confusion matrix and classification report
            all_true_labels.append(bag_label.cpu().data.numpy())
            all_pred_labels.append(predicted_label.cpu().data.numpy())

            # if batch_idx < 5:  # Plot bag labels and instance labels for the first 5 bags
            #     print(f'True Bag Label: {bag_label}, Predicted Bag Label: {predicted_label}')

    # Calculate test loss and error for epoch
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels, output_dict=True,)
    confusion_matrix_result = confusion_matrix(all_true_labels, all_pred_labels)
    auc = roc_auc_score(all_true_labels, all_pred_labels)
    print(f'Test Results:')
    print(f'Loss: {test_loss:.4f}, Test error: {test_error:.4f}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{confusion_matrix_result}')
    print(f'AUC: {auc:.4f}')

    return test_loss, test_error, report, confusion_matrix_result, auc


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        torch.cuda.empty_cache()

    print('Start Testing')
    test()
    
    '''5 independent runs for training and testing'''
    
    # accs, recalls, precs, f1s, aucs = [], [], [], [], []

    # for run in range(5):  # Repeat full train-test cycle
    #     print(f"\nRun {run+1}/5")

    #     # Reinitialize model and optimizer
    #     if args.model == 'attention':
    #         model = Attention().cuda() if args.cuda else Attention()
    #     elif args.model == 'gated_attention':
    #         model = GatedAttention().cuda() if args.cuda else GatedAttention()

    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


    #     # Train over all epochs
    #     for epoch in range(1, args.epochs + 1):
    #         train(epoch)
    #         torch.cuda.empty_cache()

    #     print('Start Testing')
    #     test_loss, test_error, report, confusion_matrix_result, auc = test()

    #     # Parse classification report and collect metrics
    #     accs.append(report['accuracy'])
    #     recalls.append(report['macro avg']['recall'])
    #     precs.append(report['macro avg']['precision'])
    #     f1s.append(report['macro avg']['f1-score'])
    #     aucs.append(auc)

    # # Print final average results
    # print("\n==== Summary over 5 runs ====")
    # print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    # print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    # print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    # print(f"F1-score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    # print(f"AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")