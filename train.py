import torch
import torchvision
import matplotlib.pyplot as plt


def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])


data = torchvision.datasets.MNIST(root="./", download=True)
show_images(data)
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 32
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
#         transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = torchvision.datasets.MNIST(root=".", download=True,
                                         transform=data_transform)
    
    train_ids, test_ids = train_test_split([i for i in range(len(dataset))], test_size=0.1, random_state=42)
    
    train = torch.utils.data.Subset(dataset, train_ids)
    
    test = torch.utils.data.Subset(dataset, test_ids)

    # test = torchvision.datasets.MNIST(root=".", download=True,
    #                                      transform=data_transform, split='test')
    return train, test


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap='gray')

    
train_dataset, test_dataset = load_transformed_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
from improved_diffusion.unet import UNetModel

model = UNetModel(
    in_channels=1, 
    model_channels=64, 
    out_channels=1,
    num_res_blocks=3, 
    attention_resolutions=("16",)
)
model
def get_loss(model, x_0, x_1, t):
    t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()
    noise_pred = model(x_t, t)
    
    return F.mse_loss(noise_pred, (x_1 - x_0), reduction="sum")
def euler_integration(net, x0, name):
    model.eval()
    
    eps = 1e-8
    n_steps = 100
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(x0.device)

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(n_steps / num_images)

    for i in range(1, len(t)):
        with torch.no_grad():
            t_prev = t[i - 1].unsqueeze(0)
#             print(x0.shape, t_prev.shape)
            f_eval = net(x0, t_prev)
        x = x0 + (t[i] - t[i - 1]) * f_eval
        x0 = x
        
        if i % stepsize == 0 or i == (len(t) - 1):
            plt.subplot(1, num_images, int(i / stepsize))
            show_tensor_image(x.detach().cpu())
    plt.savefig(name)
    return x
def train_epoch(model, dataloader, device):
    model.to(device)
    model.train()

    total_loss = 0.0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        t = torch.tensor(np.random.uniform(size=(BATCH_SIZE)), dtype=torch.float32).to(device)

        x_1 = batch[0].to(device)
        x_0 = destroy_bottom(x_1)
        
        loss = get_loss(model, x_0, x_1, t)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    for batch in tqdm(dataloader):

        t = torch.tensor(np.random.uniform(size=(BATCH_SIZE)), dtype=torch.float32).to(device)

        x_1 = batch[0].to(device)
        x_0 = destroy_bottom(x_1)
        
        with torch.no_grad():
            loss = get_loss(model, x_0, x_1, t)
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader.dataset)
from copy import deepcopy

def destroy_bottom(x):
    B, C, H, W = x.shape
    x_2 = deepcopy(x)
    noise = torch.tensor(np.random.normal(size=(B, C, H // 2, W)), dtype=torch.float32).to(x_2.device)
    x_2[:, :, H // 2:, :] = noise
    
    return x_2
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb
wandb.init(project="impainting_diffusion")

device = "cuda" if torch.cuda.is_available() else "cpu"
#model = torch.load("new_impaining_model_19.pt")
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 100 # Try more!
wandb.watch(model)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_dataloader, device)
    eval_loss = eval_epoch(model, test_dataloader, device)

    if epoch % 1 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss} Eval Loss: {eval_loss} ")
        wandb.log({"train_loss":train_loss, "eval_loss":eval_loss})
        eval_examples = next(iter(test_dataloader))[0].to(device)
        for i in range(eval_examples.shape[0]):
            euler_integration(model, destroy_bottom(eval_examples[i].unsqueeze(0)), f"fig/epoch={epoch}_example={i}.png")
            
        torch.save(model, f"new_impaining_model_{epoch + 17}.pt")