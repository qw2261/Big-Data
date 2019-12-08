from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.views import generic





from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display
import json
import time
import os
from Feature.inception_resnet_v1 import *

def camera_preprocess(img):
        img = img.convert('LA').convert('RGB').resize((48,48)).resize((160,160))
        img = torch.tensor([np.rollaxis(np.array(img)/255, 2, 0)]).float()
        return img

class Inference(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=4, dropout_prob=0.6)
        self.model.load_state_dict(torch.load('./static/SavedModel/dict.pth', map_location=torch.device('cpu')))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
    

    def predict(self, frame):
        boxes, _ = self.mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_draw = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        if boxes is not None:
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            croped = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(boxes[0])
            input_frame = camera_preprocess(croped)
            if torch.cuda.is_available():
                input_frame = input_frame.cuda()
            
            prediction = self.model.forward(input_frame).cpu().detach().numpy()[0]
            predict_lable = np.argmax(prediction)


            target = ['Angry','Happy','Neutral','Confused']

            frame =  np.array(frame_draw)[:, :, ::-1]
            frame = cv2.putText(frame, 
                            target[predict_lable]+': '+str(int(100*prediction[predict_lable]))+'%', 
                            (int(boxes[0][0]),int(boxes[0][1]-3)), 
                            cv2.FONT_HERSHEY_COMPLEX, 
                            1, 
                            (0,0,255), 
                            2)
        return frame





def hello(request):
    context = {}
    context['content1'] = 'Hello World'
    return render(request, 'helloworld.html', context)

@csrf_exempt
def goData(request):

    links = [
        {
            'Name': 'Sully with Interstellar docking scene score (No Time For Caution) HQ Edit',
            'URL': 'https://www.youtube.com/watch?v=X7iRe8O8Jsk'
        },
        {
            'Name': 'Heat Shootout Scene [High Quality]',
            'URL': 'https://www.youtube.com/watch?v=ZL9fnVtz_lc'
        } 
    ]


    return JsonResponse(links, safe=False)
    
@csrf_exempt
def dealImage(request):
    inference = Inference()
    start = time.time()

    context = {}
    context['content'] = request.method

    print('Enter to Recommend Page')
    print(request.FILES)

    if request.method == 'POST' and request.FILES:
        file_name = list(request.FILES.keys())[0]
        myfile = request.FILES[file_name]
        print("File anme: ", myfile)
        print("File type: ", type(myfile))


        
        im = Image.open(myfile)
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        im = inference.predict(im)


        cv2.imwrite("./static/img/send.png", im)

        im = Image.open("./static/img/send.png")

        print('Saved Upload Picture')

        print(time.time() - start)
        return HttpResponse(im, content_type="image/jepg")



    print(time.time() - start)
    return render(request, 'recommend.html', context)

@csrf_exempt
def identify(request):
    inference = Inference()
    start = time.time()

    context = {}
    context['content'] = request.method

    print('Enter to Recommend Page')
    print(request.FILES)

    if request.method == 'POST' and request.FILES:
        file_name = list(request.FILES.keys())[0]
        myfile = request.FILES[file_name]
        print("File anme: ", myfile)
        print("File type: ", type(myfile))

        try:
            os.remove('./static/img/real_time.png')
        except:
            pass

        
        im = Image.open(myfile)
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        im = inference.predict(im)


        cv2.imwrite("./static/img/real_time.png", im)

        print('Saved Upload Picture')



    print(time.time() - start)
    return render(request, 'identify.html', context)

@csrf_exempt
def recommend(request):
    inference = Inference()
    start = time.time()

    context = {}
    context['content'] = request.method

    print('Enter to Recommend Page')
    print(request.FILES)

    if request.method == 'POST' and request.FILES:
        file_name = list(request.FILES.keys())[0]
        myfile = request.FILES[file_name]
        print("File anme: ", myfile)
        print("File type: ", type(myfile))

        try:
            os.remove('./static/img/real_time.png')
        except:
            pass

        
        im = Image.open(myfile)
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        im = inference.predict(im)


        cv2.imwrite("./static/img/real_time.png", im)

        print('Saved Upload Picture')



    print(time.time() - start)
    return render(request, 'recommend.html', context)

@csrf_exempt
def getRecommend(request):
    context = {}
    context['content1'] = 'Hello World'
    return render(request, 'getRecommend.html', context)