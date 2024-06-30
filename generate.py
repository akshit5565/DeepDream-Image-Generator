import torch
import torchvision.utils as vutils
from models import Generator
from config import *
import torchvision.transforms as transforms
from PIL import Image

def generate_images(input_image_path,output_image_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load('generator.pth', map_location=device))
    generator.eval()
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    input_features = input_tensor.mean(dim=(2, 3)).squeeze()
    
    if input_features.numel() < latent_dim:
        padding = torch.zeros(latent_dim - input_features.numel(), device=device)
        input_features = torch.cat((input_features, padding))
    else:
        input_features = input_features[:latent_dim]
    
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        
        input_features = input_features.view(1, latent_dim, 1, 1)
        modified_noise = noise + input_features
        
        fake_images = generator(modified_noise)

    output_image = transforms.Resize(output_image_size)(fake_images.squeeze(0).cpu())
    output_image_path = 'app/static/generated/output.png'  
    vutils.save_image(output_image, output_image_path, normalize=True)

    return output_image_path