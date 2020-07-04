import torch
import modules
import torch.nn.functional as F
import torchvision
import os

def generate_bw(net, device, image_path, batch_size=32, image_size=28):
    """
    Generate single channel image
    """
    net.eval()
    img = torch.Tensor(batch_size, 1, image_size, image_size).to(device)
    img.fill_(0)

    for i in range(image_size):
        for j in range(image_size):
            out = net(img)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            img[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

    #Saving images row wise
    torchvision.utils.save_image(img, image_path, nrow=12, padding=0)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = modules.PixelCNN()
    net = net.to(device)

    net.load_state_dict(torch.load('../output/model.pt'))
    generate_bw(net, device, '../output/sample.png')
