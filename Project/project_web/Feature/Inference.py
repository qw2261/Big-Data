import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display


def camera_preprocess(img):
    img = img.convert('LA').convert('RGB').resize((48,48)).resize((160,160))
    img = torch.tensor([np.rollaxis(np.array(img)/255, 2, 0)]).float()
    return img

def inference(img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(keep_all=True, device=device)


    model = torch.load('./static/SavedModel/test.pth').cuda().eval()


    boxes, _ = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    frame_draw = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
    if boxes is not None:
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        croped = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(boxes[0])
        input_frame = camera_preprocess(croped)


        
        prediction = model.forward(input_frame.cuda()).cpu().detach().numpy()[0]
        predict_lable = np.argmax(prediction)


        target = ['Angry','Happy','Neutral','Confused']

        img =  np.array(frame_draw)[:, :, ::-1]
        img = cv2.putText(img, 
                        target[predict_lable]+': '+str(int(100*prediction[predict_lable]))+'%', 
                        (int(boxes[0][0]),int(boxes[0][1]-3)), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        1, 
                        (0,0,255), 
                        2)
        return img


