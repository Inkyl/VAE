import matplotlib.pyplot as plt
import torch

from utils import VAE


def plot_interpolations(model, device, z_dim=64, num_interpolations=10):
    # 假设 z1 和 z2 是你的两个随机噪声向量
    z1 = torch.randn((1, z_dim)).to(device)
    z2 = torch.randn((1, z_dim)).to(device)

    # 对 z1 和 z2 进行线性插值
    interpolated_images = []
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)  # 插值参数
        z = alpha * z1 + (1 - alpha) * z2  # 进行线性插值
        interpolated_image = model.decoder(z)  # 生成插值后的图像
        interpolated_images.append(interpolated_image)

    # 将所有的插值图像合并在一起
    interpolated_images = torch.cat(interpolated_images, dim=0)

    # 将插值后的图像 reshape 成适当的形状
    interpolated_images = interpolated_images.view(interpolated_images.shape[0], 28, 28)

    fig, axs = plt.subplots(1, num_interpolations, figsize=(15, 2))
    for i, img in enumerate(interpolated_images):
        axs[i].imshow(img.detach().cpu().numpy(), cmap='gray')
        axs[i].axis('off')
    plt.savefig('interpolations.png')


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = VAE(28 * 28, 28 * 28, 256, 64).to(device)
checkpoint = torch.load('output/vae.pth')
model.load_state_dict(checkpoint['model'])
print(checkpoint['epoch'])
plot_interpolations(model, device,num_interpolations=20)
