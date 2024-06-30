# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms #used for data preprocessing and augmentation
# import torchvision.utils as vutils # useful for visualizing and saving images
# from config import *
# from models import Generator, Discriminator

# results_dir = 'C:\\Users\\Akshit\\Desktop\\DeepDream-Image-Generator\\results'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir) 

# real_samples_path = os.path.join(results_dir, 'real_samples.png')

# torch.manual_seed(manual_seed)

# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(), # Converts the image to a PyTorch tensor and scales the pixel values to the range [0, 1]
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # Normalizes the tensor with mean 0.5 and standard deviation 0.5 for each of the three channels (R, G, B), scaling the pixel values to the range [-1, 1]
# ])

# dataset = dsets.CIFAR10(root='./data', download=True, transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generator = Generator(latent_dim).to(device)
# discriminator = Discriminator().to(device)

# criterion = nn.BCELoss() #binary cross entropy 
# optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas = (beta1,0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1,0.999))

# if __name__ == "__main__":
#     # Training loop
#     for epoch in range(num_epochs):
#         for i, data in enumerate(dataloader, 0):
#             discriminator.zero_grad()  # Clears the gradients of the discriminator

#             real_images, _ = data  # Unpacking real_images and corresponding labels, labels are ignored here
#             real_images = real_images.to(device)
#             batch_size = real_images.size(0)
#             label = torch.full((batch_size,), 1, device=device, dtype=torch.float)  # Creates a tensor of labels filled with ones (indicating real images) with the same batch size
#             output = discriminator(real_images).view(-1)  # Feeds the real images into the discriminator and flattens the output
#             errD_real = criterion(output, label)
#             errD_real.backward()  # Backpropagates the loss for real images
#             D_x = output.mean().item()  # Computes the mean of the discriminator's output for real images

#             noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # Generates random noise 
#             fake_images = generator(noise)  # Feeds noise into generator to produce fake images
#             label.fill_(0.0)  # Fills the label tensors with zeros indicating fake images
#             output = discriminator(fake_images.detach()).view(-1)  # Feeds fake images into discriminator and flattens the output
#             errD_fake = criterion(output, label)  # Computes BCE for fake images
#             errD_fake.backward()
#             D_G_z1 = output.mean().item()  # Computes mean of the discriminator's output for fake images

#             errD = errD_fake + errD_real  # Total discriminator loss
#             optimizer_D.step()  # Update discriminator's parameters

#             generator.zero_grad()  # Clear generator's gradients
#             label.fill_(1.0)  # Fills the label tensors with ones indicating real images
#             output = discriminator(fake_images).view(-1)
#             errG = criterion(output, label)
#             errG.backward()
#             D_G_z2 = output.mean().item()
#             optimizer_G.step()

#             if i % 100 == 0:
#                 print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
#                       % (epoch, num_epochs, i, len(dataloader),
#                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#         # Save generated images
#         # Save generated images
#         if epoch == 0:
#             vutils.save_image(real_images, '%s/real_samples.png' % results_dir, normalize=True)
#         fake = generator(noise)
#         vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (results_dir, epoch), normalize=True)


#     # Save the trained model
#     torch.save(generator.state_dict(), 'generator.pth')



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
from config import *
from models import Generator, Discriminator

results_dir = 'C:\\Users\\Akshit\\Desktop\\DeepDream-Image-Generator\\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

torch.manual_seed(manual_seed)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load custom dataset from folder 'animeimages'
dataset = dsets.ImageFolder(root='animeimages', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()

            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), 1, device=device, dtype=torch.float)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(0.0)
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_fake + errD_real
            optimizer_D.step()

            generator.zero_grad()
            label.fill_(1.0)
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save generated images
        if epoch == 0:
            vutils.save_image(real_images, '%s/real_samples.png' % results_dir, normalize=True)
        fake = generator(noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (results_dir, epoch), normalize=True)

    # Save the trained model
    torch.save(generator.state_dict(), 'generator.pth')
