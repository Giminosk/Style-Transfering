import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import json
from model import VGG


with open('./config.json', 'r') as f:
    config = json.load(f)
# hyperparameters
imsize = config['image_size']
total_steps = config['total_steps']
learning_rate = config['learning_rate']
alpha = config['alpha']
beta = config['beta']
# means and stds originally used for training VGG19 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def image_loader(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    return img


def train():
    original_img = image_loader(config['original_image_path'])
    style_img = image_loader(config['style_image_path'])
    input_img = original_img.clone().requires_grad_(True)
    
    model = VGG(pretrained=False, weights_path="./vgg19/vgg19_pretrained.pth.tar")
    model = model.to(device).eval()
    optimizer = torch.optim.Adam([input_img], lr=learning_rate)
    
    for step in range(total_steps):
        
        input_img_features = model(input_img)
        original_img_features = model(original_img)
        style_img_features = model(style_img)
        features = zip(input_img_features, original_img_features, style_img_features)

        style_loss = 0
        original_loss = 0

        for input_feature, orig_feature, style_feature in features:
            
            original_loss += torch.mean((input_feature - orig_feature) ** 2)
            
            a, b, c, d = input_feature.shape
            gram_mat_gen = input_feature.view(a*b, c*d).mm(input_feature.view(a*b, c*d).t())
            gram_mat_style = style_feature.view(a*b, c*d).mm(style_feature.view(a*b, c*d).t())
            style_loss += torch.mean((gram_mat_gen - gram_mat_style) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'==> Step {step}, Total Loss: {total_loss.item()}, Original Loss: {original_loss.item()}, Style Loss: {style_loss.item()}')

    torchvision.utils.save_image(inv_transform(input_img), config['result_path'])
    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

inv_transform = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)


if __name__ == "__main__":
    train()