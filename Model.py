import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),  # Input channels changed to 3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # Dynamically calculate the input size for the linear layer
        self.conv_output_size = None  # Will be set during forward pass
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),  # Placeholder, updated dynamically
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # Matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # Matrix w
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the Attention model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, channels, height, width].
        Returns:
            tuple: A tuple (Y_prob, Y_hat, A), where:
                - Y_prob: Probability of the positive class.
                - Y_hat: Predicted label (0 or 1).
                - A: Attention weights.
        """
        # x shape: [batch_size, num_patches, channels, height, width]
        batch_size, num_patches, channels, height, width = x.shape

        # Flatten the batch and patch dimensions to process each patch independently
        x = x.view(batch_size * num_patches, channels, height, width)

        # print(torch.cuda.memory_summary())
        # Feature extraction (part 1)
        H = self.feature_extractor_part1(x)
        print(f"Feature maps shape after part 1: {H.shape}")
        # Dynamically calculate the output size of the convolutional layers
        if self.conv_output_size is None:
            self.conv_output_size = torch.prod(torch.tensor(H.shape[1:])).item()
            print(f"Convolutional output size: {self.conv_output_size}")
            # Update the linear layer in feature_extractor_part2
            self.feature_extractor_part2[0] = nn.Linear(self.conv_output_size, self.M).to(H.device)


        # Flatten the feature maps
        H = H.view(-1, self.conv_output_size)

        # Feature extraction (part 2)
        H = self.feature_extractor_part2(H)  # Shape: [batch_size * num_patches, M]
        print(f"Feature maps shape after dynamic update: {H.shape}")

        # Reshape back to [batch_size, num_patches, M]
        H = H.view(batch_size, num_patches, -1)

        # Compute attention scores
        A = self.attention(H)  # Shape: [batch_size, num_patches, ATTENTION_BRANCHES]
        A = torch.transpose(A, 1, 2)  # Shape: [batch_size, ATTENTION_BRANCHES, num_patches]
        A = F.softmax(A, dim=2)  # Softmax over patches

        # Weighted sum of features
        Z = torch.matmul(A, H)  # Shape: [batch_size, ATTENTION_BRANCHES, M]

        # Classification
        Y_prob = self.classifier(Z.squeeze(1))  # Remove ATTENTION_BRANCHES dimension
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        """
        Calculate the classification error.
        Args:
            X (torch.Tensor): Input tensor.
            Y (torch.Tensor): Ground truth labels.
        Returns:
            tuple: A tuple (error, Y_hat), where:
                - error: Classification error rate.
                - Y_hat: Predicted labels.
        """
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        """
        Calculate the objective function (negative log likelihood).
        Args:
            X (torch.Tensor): Input tensor.
            Y (torch.Tensor): Ground truth labels.
        Returns:
            tuple: A tuple (neg_log_likelihood, A), where:
                - neg_log_likelihood: Negative log likelihood loss (scalar).
                - A: Attention weights.
        """
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)  # Clamp probabilities to avoid numerical instability
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        neg_log_likelihood = neg_log_likelihood.mean()  # Reduce to a scalar by averaging
        return neg_log_likelihood, A
    
class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),  # Input channels changed to 3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # Dynamically calculate the input size for the linear layer
        self.conv_output_size = None  # Will be set during forward pass
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),  # Placeholder, updated dynamically
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the GatedAttention model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, channels, height, width].
        Returns:
            tuple: A tuple (Y_prob, Y_hat, A), where:
                - Y_prob: Probability of the positive class.
                - Y_hat: Predicted label (0 or 1).
                - A: Attention weights.
        """
        # x shape: [batch_size, num_patches, channels, height, width]
        batch_size, num_patches, channels, height, width = x.shape

        # Flatten the batch and patch dimensions to process each patch independently
        x = x.view(batch_size * num_patches, channels, height, width)

        # Feature extraction (part 1)
        H = self.feature_extractor_part1(x)

        # Dynamically calculate the output size of the convolutional layers
        if self.conv_output_size is None:
            self.conv_output_size = torch.prod(torch.tensor(H.shape[1:])).item()
            print(f"Convolutional output size: {self.conv_output_size}")
            # Update the linear layer in feature_extractor_part2
            self.feature_extractor_part2[0] = nn.Linear(self.conv_output_size, self.M).to(H.device)

        # Flatten the feature maps
        H = H.view(-1, self.conv_output_size)

        # Feature extraction (part 2)
        H = self.feature_extractor_part2(H)  # Shape: [batch_size * num_patches, M]

        # Reshape back to [batch_size, num_patches, M]
        H = H.view(batch_size, num_patches, -1)

        # Compute attention scores
        A_V = self.attention_V(H)  # Shape: [batch_size, num_patches, L]
        A_U = self.attention_U(H)  # Shape: [batch_size, num_patches, L]
        A = self.attention_w(A_V * A_U)  # Element-wise multiplication
        A = torch.transpose(A, 1, 2)  # Shape: [batch_size, ATTENTION_BRANCHES, num_patches]
        A = F.softmax(A, dim=2)  # Softmax over patches

        # Weighted sum of features
        Z = torch.matmul(A, H)  # Shape: [batch_size, ATTENTION_BRANCHES, M]

        # Classification
        Y_prob = self.classifier(Z.squeeze(1))  # Remove ATTENTION_BRANCHES dimension
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        """
        Calculate the classification error.
        Args:
            X (torch.Tensor): Input tensor.
            Y (torch.Tensor): Ground truth labels.
        Returns:
            tuple: A tuple (error, Y_hat), where:
                - error: Classification error rate.
                - Y_hat: Predicted labels.
        """
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        """
        Calculate the objective function (negative log likelihood).
        Args:
            X (torch.Tensor): Input tensor.
            Y (torch.Tensor): Ground truth labels.
        Returns:
            tuple: A tuple (neg_log_likelihood, A), where:
                - neg_log_likelihood: Negative log likelihood loss (scalar).
                - A: Attention weights.
        """
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)  # Clamp probabilities to avoid numerical instability
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        neg_log_likelihood = neg_log_likelihood.mean()  # Reduce to a scalar by averaging
        return neg_log_likelihood, A