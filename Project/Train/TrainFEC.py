import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch.utils import data
from torch.nn.modules.distance import PairwiseDistance
import gc
from tqdm import tqdm
import cv2

from sklearn.model_selection import KFold
from inception_resnet_v1 import *
from eval_metrics import *

import datas


class DNNTrain(object):
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.TripletMarginLoss(margin=0.2)
        self.l2_dist = PairwiseDistance(2)


    def train(self, loader, num_epoch):
        epoch = 0
        last_loss = 100
        for _ in range(num_epoch):
            epoch +=1
            print('epoch:', epoch)
            gc.collect()
            valid_loss = self.train_epoch(loader['train'], loader['test'])
            if last_loss > valid_loss:
                # torch.save(self.network, './SavedModel/triplet.pth')
                torch.save(self.network.state_dict(), './SavedModel/triplet_dense.pth')
                last_loss = valid_loss
            else:
                continue

    
    def train_epoch(self, train_loader, test_loader):
        self.network.train()
        labels, distances = [], []
        triplet_loss_sum = 0.0
        for i, (anc, pos, neg) in enumerate(tqdm(train_loader)):
            if torch.cuda.is_available():
                anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()
                self.network.cuda()
            self.optimizer.zero_grad()
            anc, pos, neg = Variable(anc), Variable(pos), Variable(neg)
            anc_fea = self.network(anc)
            pos_fea = self.network(pos)
            neg_fea = self.network(neg)

            loss = self.criterion(anc_fea, pos_fea, neg_fea)
            loss.backward()
            self.optimizer.step()

            dists = self.l2_dist.forward(anc_fea, pos_fea)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0)))

            dists = self.l2_dist.forward(anc_fea, neg_fea)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))
            triplet_loss_sum += loss.item()


        avg_triplet_loss = triplet_loss_sum / trainset.__len__()
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        print(labels)
        print(distances)
        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  train set - Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
        print('  train set - Accuracy           = {:.8f}'.format(np.mean(accuracy)))
            
        with torch.no_grad():
            self.network.eval()
            labels, distances = [], []
            triplet_loss_sum = 0.0
            for i, (anc, pos, neg) in enumerate(tqdm(test_loader)):
                if torch.cuda.is_available():
                    anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()
                    self.network.cuda()
                anc, pos, neg = Variable(anc), Variable(pos), Variable(neg)
                anc_fea = self.network(anc)
                pos_fea = self.network(pos)
                neg_fea = self.network(neg)

                loss = self.criterion(anc_fea, pos_fea, neg_fea)

                dists = self.l2_dist.forward(anc_fea, pos_fea)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.ones(dists.size(0)))

                dists = self.l2_dist.forward(anc_fea, neg_fea)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.zeros(dists.size(0)))
                triplet_loss_sum += loss.item()


            avg_triplet_loss = triplet_loss_sum / testset.__len__()
            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            print(labels)
            print(distances)
            tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
            print('  test set - Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
            print('  test set - Accuracy           = {:.8f}'.format(np.mean(accuracy)))
            return avg_triplet_loss



if __name__ == "__main__":
    path = './Data/Team'
    
    transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()])

    trainset = datas.fec_data.FecData(transform)
    testset = datas.fec_data.FecTestData(transform)
    trainloader = data.DataLoader(trainset, batch_size=24, num_workers=16)
    testloader = data.DataLoader(testset, batch_size=24, num_workers=16)

    data_loader = {'train': trainloader, 'test': testloader}

    # model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=16, dropout_prob=0.6)
    # model.load_state_dict(torch.load('./SavedModel/triplet2.pth'))
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 16)
    print(model)

    trainer = DNNTrain(model, 1e-4)
    trainer.train(data_loader, 50)\