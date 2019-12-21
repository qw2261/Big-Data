import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import gc
from tqdm import tqdm
import cv2

from inception_resnet_v1 import *


class DNNTrain(object):
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()


    def train(self, loader, num_epoch):
        epoch = 0
        last_loss = 100
        for _ in range(num_epoch):
            epoch +=1
            print('epoch:', epoch)
            gc.collect()
            valid_loss = self.train_epoch(loader['train'], loader['validation'])
            if last_loss >= valid_loss:
                torch.save(self.network, './SavedModel/test_team.pth')
            else:
                continue

    
    def train_epoch(self, train_loader, valid_loader):
        self.network.train()
        total_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = Variable(images)
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                self.network.cuda()
            self.optimizer.zero_grad()
            predictions = self.network(images)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() 
        print('Train loss=', total_loss/i)
            
        with torch.no_grad():
            self.network.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            for i, (images, labels) in enumerate(valid_loader):
                images_Var = Variable(images)
                labels_Var = Variable(labels)
                if torch.cuda.is_available():
                    images_Var = images_Var.cuda()
                    labels_Var = labels_Var.cuda()
                    self.network.cuda()
                predictions = self.network(images_Var)
                loss = self.criterion(predictions, labels_Var)
                valid_loss += loss.item()
                
                predict_lable = np.argmax(predictions.cpu().numpy(), axis = 1)
                valid_acc += sum(predict_lable == labels.numpy())/len(predict_lable)
            valid_loss /= len(valid_loader) 
            valid_acc /= len(valid_loader)
            print('Validation accuracy = ', valid_acc, 'Validation loss = ', valid_loss)
        return valid_loss




if __name__ == "__main__":
    path = './Data/Images4c'
    transform = transforms.Compose([transforms.Resize(160), transforms.ToTensor()])
    data_image = {x:datasets.ImageFolder(root = os.path.join(path,x), transform = transform) for x in ['train', 'test']}

    index = list(range(len(data_image['train'])))    
    random.shuffle(index)           
    train_loader = torch.utils.data.DataLoader(data_image['train'], batch_size=100, sampler=SubsetRandomSampler(index[2000:]))
    valid_loader = torch.utils.data.DataLoader(data_image['train'], batch_size=100, sampler=SubsetRandomSampler(index[:2000]))
    test_loader = torch.utils.data.DataLoader(data_image['test'], batch_size=100, shuffle=True)
    data_loader = {'train': train_loader, 'validation': valid_loader, 'test': train_loader}

    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=4, dropout_prob=0.6)
    model = torch.load('./SavedModel/test.pth')
    print(model)

    trainer = DNNTrain(model, 1e-4)
    trainer.train(data_loader, 50)