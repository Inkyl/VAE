import torch
import torch.nn as nn
import argparse
import os

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from utils import VAE


def train():
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(os.path.join(output_dir, log_dir))
    total_step = 0
    mini_loss = 100000
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (data, _) in enumerate(train_dataloader):
            data = data.view(data.shape[0], -1)
            data = data.to(device)
            optimizer.zero_grad()
            predict, mu, logvar = model(data)
            reconstruction_loss = loss_fn(predict, data)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f'epoch: {epoch}, step: {i}, loss: {loss.item()}')
                writer.add_scalar('Step_Loss', loss.item(), total_step)
                writer.add_scalar('KL_Loss', kl_divergence.item(), total_step)
                writer.add_scalar('Reconstruction_Loss', reconstruction_loss.item(), total_step)
                total_step += 1
        print(f'epoch: {epoch} , loss: {epoch_loss / i}')
        if epoch_loss / i < mini_loss:
            mini_loss = epoch_loss
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(output_dir, 'vae.pth'))
        with torch.no_grad():
            z = torch.randn((batch_size, 64)).to(device)
            samples = model.decoder(z)
            samples = samples.view(samples.shape[0], 1, 28, 28)
            grid = make_grid(samples).unsqueeze(0)
            writer.add_images('Generated_Images', grid, epoch)
        writer.add_scalar('Epoch_Loss', epoch_loss / i, epoch)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='data')
    parse.add_argument('--output_dir', type=str, default='./output')
    parse.add_argument('--log_dir', type=str, default='log')
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--num_workers', type=int, default=4)
    parse.add_argument('--num_epochs', type=int, default=100)
    parse.add_argument('--lr', type=float, default=0.001)
    args = parse.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    log_dir = args.log_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    lr = args.lr

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    train_dataset = torchvision.datasets.MNIST(root='../../datasets/torch', train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    model = VAE(28 * 28, 28 * 28, 256, 64).to(device)
    train()
