import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# reference : https://mvje.tistory.com/206

class Encoder(nn.Module):
    def __init__(self, input_dim, hiddend_dim, latent_dim, kenersize=(4,4,3,1), stride=2):
        super(Encoder,self).__init__()

        kerner_1,kerner_2,kerner_3,kerner_4 = kenersize

        self.stride_conv1 = nn.Conv2d(input_dim, hiddend_dim, kerner_1, stride, padding=1)
        self.stride_conv2 = nn.Conv2d(hiddend_dim, hiddend_dim, kerner_2, stride, padding=1)

        self.residual_conv1 = nn.Conv2d(hiddend_dim, hiddend_dim, kerner_3, padding=1)
        self.residual_conv2 = nn.Conv2d(hiddend_dim, hiddend_dim, kerner_4, padding=0)

        self.latent = nn.Conv2d(hiddend_dim, latent_dim * 2, kernel_size=1)

        self.latent_dim = latent_dim

    def forward(self,x):
        x = self.stride_conv1(x)
        x = self.stride_conv2(x)
        x = F.relu(x)

        residual_x = self.residual_conv1(x)
        residual_x = residual_x+x # Weighted sum

        x = F.relu(residual_x)
        residual_x = self.residual_conv2(x)
        residual_x = residual_x+x # Weighted sum

        residual_x = self.latent(residual_x)
        
        mu, logvar = residual_x[:, :self.latent_dim], residual_x[:, self.latent_dim:]
        return residual_x, mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim, hiddend_dim, latent_dim, kenersize=(1,3,2,2), stride=2):
        super(Decoder,self).__init__()

        kerner_1,kerner_2,kerner_3,kerner_4 = kenersize # 중요

        self.latent = nn.Conv2d(latent_dim, hiddend_dim, kernel_size=1)

        self.residual_conv1 = nn.Conv2d(hiddend_dim, hiddend_dim, kerner_1, padding=0)
        self.residual_conv2 = nn.Conv2d(hiddend_dim, hiddend_dim, kerner_2, padding=1)
        
        self.stride_conv1 = nn.ConvTranspose2d(hiddend_dim, hiddend_dim, kerner_3, stride, padding=0) # Upsampling 을 하기 위해 변경된 Conv2D
        self.stride_conv2 = nn.ConvTranspose2d(hiddend_dim, input_dim, kerner_4, stride, padding=0) # Upsampling 을 하기 위해 변경된 Conv2D
        

        self.latent_dim = latent_dim

    def forward(self,x):
        x = self.latent(x)

        residual_x = self.residual_conv1(x)
        residual_x = residual_x+x
        x = F.relu(residual_x)

        residual_x = self.residual_conv2(x)
        residual_x = residual_x+x
        x = F.relu(residual_x)

        x = self.stride_conv1(x)
        x = self.stride_conv2(x)

        return x

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # latent vectr를 샘플링
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # 로그 분산으로 표준편차를 계산
        eps = torch.randn_like(std) # 표준 정규 분포에서 샘플린된 랜덤노이즈 생성 **중요**
        return mu + eps * std

    def forward(self, x):
        # 인코더를 통해 mu와 logvar 계산
        enc_output, mu, logvar = self.encoder(x)

        # 리파라미터화 트릭을 사용하여 잠재 변수 샘플링
        z = self.reparameterize(mu, logvar)
        # print(z)
        # 디코더를 통해 잠재 변수로부터 이미지 생성
        recon_x = self.decoder(z)

        return recon_x, mu, logvar
    
def main():

    dataset_path = '~/datasets'
    batch_size = 128

    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 
    
    # train, test dataset 설정
    train_dataset = CIFAR10(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = CIFAR10(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,  **kwargs)
    

    encoder = Encoder(input_dim=3,hiddend_dim=32, latent_dim=16).to(DEVICE)
    decoder = Decoder(input_dim=3,hiddend_dim=32, latent_dim=16).to(DEVICE)
    # model 설정
    model = VAE(encoder,decoder).to(DEVICE)
    # learning rate 설정
    lr = 2e-4

    # optimizer 설정
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    bce_loss = nn.MSELoss().to(DEVICE)
    print_step = 50
    epochs = 10000
    for epoch in range(epochs):
        for bactch_idx, (x,_) in enumerate(train_loader):
            # print(DEVICE)
            # x = x.detach().to(DEVICE)
            x = x.to(DEVICE)

            # optimizer 초기화
            optimizer.zero_grad()

            recon_x, mu, logvar = model(x)

            # Loss 계산
            BCE = bce_loss(recon_x, x)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = BCE + KLD

            # 훈련을 위한 셋업
            loss.backward()
            optimizer.step()
            if bactch_idx % print_step ==0:
                print(r'epoch:{0}, recon loss:{1}, KLD loss:{2}, total loss:{3}'.format(epoch, BCE, KLD, loss))

    print("finish")



if __name__=='__main__':
    main()
