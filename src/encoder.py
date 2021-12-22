"""Creates an Encoder CNN for the agent to predict actions given a state."""

import pdb

import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, action_space_size=4) -> None:
        """Initializes the Encoder Neural Network.
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 256, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, action_space_size)

        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.relu = nn.ReLU()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """Feeds the input tensor forward through the CNN Encoder.

        Args:
            x (torch.Tensor): Batch of RGB Images, shape (N,C,H,W).
                N: batch size
                C: 3
                H: image height
                W: image width
        Returns:
            output: Q-values for each input image in the batch, shape (N, action_space_size)
        """
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.flatten(x) 
        x = self.relu(self.linear1(x))
        output = self.relu(self.linear2(x))

        return output

if __name__ == "__main__":
    encoder = Encoder()

    input = torch.rand(64, 3, 10, 10)
    output = encoder.forward(input)
    print(output.shape)