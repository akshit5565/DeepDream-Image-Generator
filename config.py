import os

manual_seed = 42
batch_size = 128
image_size = 32
latent_dim = 100
num_epochs = 1
lr = 0.0002
beta1 = 0.5

results_dir = './results'
generated_samples_path = './results/generated_samples.png'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)