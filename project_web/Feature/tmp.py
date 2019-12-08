import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display

from inception_resnet_v1 import *



model = torch.load('./SavedModel/test_team.pth')
torch.save(model.state_dict(), './SavedModel/test.pt')




model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=4, dropout_prob=0.6)

model.load_state_dict(torch.load('./SavedModel/test.pt'))