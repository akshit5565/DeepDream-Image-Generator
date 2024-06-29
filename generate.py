import torch
import torchvision.utils as vutils
from models import Generator
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim).to(device)
generator_path = 'C:\\Users\\Akshit\\Desktop\\DeepDream-Image-Generator\\generator.pth'

generator.load_state_dict(torch.load(generator_path, map_location=device)) #loads the pre-trained weights of the generator from the specified file, map_location ensures that the model weightssa re loaded onto the correct device(CPU or GPU)
generator.eval() #seta the generator to evaluation mode, it disables certain layers like dropout that are only used during training

with torch.no_grad(): #disables gradient calculation
    noise = torch.randn(batch_size,latent_dim,1,1,device=device) #generates a batch of random noise vectors each of size (latent_dim,1,1)
    fake_images = generator(noise)

vutils.save_image(fake_images, generated_samples_path, normalize=True)
print(f"Generated images saved at '{generated_samples_path}'.")