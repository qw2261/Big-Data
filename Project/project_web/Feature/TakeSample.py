import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import glob
from PIL import Image, ImageDraw
from IPython import display

from inception_resnet_v1 import *

def camera_preprocess(img):
    img = img.convert('LA').convert('RGB').resize((48,48))
    img = torch.tensor([np.rollaxis(np.array(img)/255, 2, 0)]).float()
    return img

def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

    cv2.namedWindow("croped")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        
    savePath = './Data/Team/sad/'

    
    list_of_files = glob.glob(savePath + '*')
    if len(list_of_files):
        latest_file = max(list_of_files, key=os.path.getctime)
        i = int(latest_file[:-4].replace(savePath, ''))+1
    else:
        i = 0

    while rval:
        boxes, _ = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_draw = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        if boxes is not None:
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            croped = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(boxes[0])
            input_frame = camera_preprocess(croped)

            img = np.rollaxis(input_frame[0].numpy(), 0, 3)
            cv2.imshow("croped",  img)
            cv2.imwrite(savePath+str(i)+'.jpg',img*255)
            i += 1
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()