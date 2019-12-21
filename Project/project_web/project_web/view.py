from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.views import generic

from facenet_pytorch import MTCNN, InceptionResnetV1 as Irv1
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display
import json
import time
import os
from Feature.inception_resnet_v1 import *
import requests as req
import datetime

from scipy.spatial.distance import pdist, squareform, cosine, euclidean
import pandas as pd


import requests
from torchvision import models


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

import urllib.request
import urllib

from google.cloud import bigquery, storage

def storeBucket(name,path):
    bucket_name = "dataproc-fd9fb6e9-a8af-42d8-84df-f465b946566c-us-central1"
    storage_client = storage.Client.from_service_account_json(
            './static/img/credentials.json')
    bucket = storage_client.get_bucket(bucket_name)
    blob = storage.Blob(name, bucket)
    blob.upload_from_filename(path)
def queryBQ(query):
    client = bigquery.Client.from_service_account_json('./static/img/credentials.json')
    res = list(client.query(query).result())
    if len(res) == 0:
        return -1
    else:
        return res

def get_title(VideoID):
    APIKEY = '' # provide API key

    params = {'id': VideoID, 'key': APIKEY,
            'fields': 'items(id,snippet(channelId,title,categoryId),statistics)',
            'part': 'snippet,statistics'}

    url = 'https://www.googleapis.com/youtube/v3/videos'

    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        return data['items'][0]['snippet']['title']

def CollaborativeFiltering(ratings_path='./static/img/ratings.csv',num_to_recommend=3,subset_users=3):
    """
    Collaborative filtering based on implicit preferences
    Args: ratings_path: path to ratings.csv. num_to_recommend: number of items to recommend to each users. subset_users: don't touch
    Returns: A Spark Dataframe with num_to_recommend item IDs for each user.
    Usage: CollaborativeFiltering('ratings.csv',5).show() will display the top 5 recommendations for each user
    The format of the recommendation is a list of pairs [ID,rating] where ID is the recommended item ID and rating is the predicted rating
    """
    spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

    ratings = spark.read.csv(ratings_path,header=True)
    ratings = ratings.drop("_c0")
    ratings = ratings.withColumn("user",col("user").cast(IntegerType())).withColumn("item",col("item").cast(IntegerType())).withColumn("rating",col("rating").cast(IntegerType()))
    (training, test) = ratings.randomSplit([0.8, 0.2])
    als = ALS(maxIter=5, regParam=0.01,implicitPrefs=True, userCol="user", itemCol="item", ratingCol="rating",
          coldStartStrategy="drop")
    model = als.fit(training)
    """
    commenting out test evaluation metric because we don't have enough data yet for it to be meaningful
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    """
    userRecs = model.recommendForAllUsers(num_to_recommend)
    # itemRecs = model.recommendForAllItems(num_to_recommend)
    # users = ratings.select(als.getUserCol()).distinct().limit(subset_users)
    # userSubsetRecs = model.recommendForUserSubset(users, num_to_recommend)
    # videos = ratings.select(als.getItemCol()).distinct().limit(subset_users)
    # videoSubsetRecs = model.recommendForItemSubset(videos, num_to_recommend)
    # userRecs.show()
    return userRecs


global CFTable
print('Reading CFTable')
CFTable = CollaborativeFiltering()
print('CFTable Loaded')



global emotion_label
emotion_label = 0

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
        global emotion_label
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
            emotion_label = predict_lable


            target = ['Angry','Happy','Neutral','Confused']

            frame =  np.array(frame_draw)[:, :, ::-1]
            frame = cv2.putText(frame,
                            target[predict_lable]+': '+str(int(100*prediction[predict_lable]))+'%',
                            (int(boxes[0][0]),int(boxes[0][1]-3)),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0,0,255),
                            2)
        return frame

####################################################


class current_user(object):
    def __init__(self, user, item):
        self.user = user
        self.item = item
portfolio = current_user(None, None)



def preprocess(path):
    pic = Image.open(path)
    x = (np.rollaxis(np.array(pic)/255,2,0))
    x = x[None]
    x = torch.Tensor(x).float()
    return x

def recognize(threshold = 0.5):
    resnet = Irv1(pretrained='vggface2').eval()
    img = preprocess("./static/img/real_time.png")
    user_embedding = resnet.forward(img).detach().numpy()[0].squeeze()
    with open("./static/img/users.json", "r") as f:
        data = json.load(f)
    users = {id:np.asarray(data[id]) for id in data.keys()}
    distances = [euclidean(user_embedding,users[id]) for id in users.keys()]
    identified = False
    print(distances)
    for i,d in enumerate(distances):
        if d < threshold:
            identified = True
            id = i
            users[id] = user_embedding
            break
    if not identified:
        print("not identified")
        id = len(users)
        users[id] = user_embedding
        for id in users.keys():
            users[id] = users[id].tolist()
        with open("./static/img/users.json", "w") as file:
            json.dump(users,file)
    return id
############################################################
def FEC():
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 16)
    model.load_state_dict(torch.load("./static/SavedModel/triplet_dense.pth",map_location=torch.device("cpu")))
    pic = "./static/img/real_time.png"
    return model.forward(preprocess(pic)).detach().numpy()[0].squeeze()

############################################################

def update_recommend(id,url,expression,threshold = 0.9):
    ratings = pd.read_csv("./static/img/ratings.csv",index_col=0)
    with open("./static/img/items.json","r") as f:
        items = json.load(f)
    if url not in items:
        items[url] = len(items)
    item = items[url]
    with open("./static/img/items.json","w") as f:
        json.dump(items,f)

    clusters = [[-0.18853393,  0.09626842,  0.2765461 , -0.07538239, -0.14150989,
        0.11746827,  0.03892784,  0.03327456,  0.2195847 , -0.12514782,
        0.26053393, -0.09284508, -0.04413346,  0.06036556,  0.3372671 ,
       -0.12304152]]
    dist = min(euclidean(expression,cluster) for cluster in clusters)
    if dist < threshold:
        implicit_rating = 1
    else:
        implicit_rating = 0

    if not id in ratings['user']:
        print("new user rating update")
        data = {"user":id,"item":item,"rating":implicit_rating}
        temp_df = pd.DataFrame(data = data,index=[0])
        ratings = ratings.append(temp_df,ignore_index=True)
    else:
        if ratings.loc[(ratings['user'] == id) & (ratings['item'] == item),"rating"].empty:
            print(id)
            print(item)
            print("new item for user")
            data = {"user":id,"item":item,"rating":implicit_rating}
            ratings = ratings.append(pd.DataFrame(data=data,index=[0]),ignore_index=True)
        else:
            print("updating rating for item")
            ratings.loc[(ratings['user'] == id) & (ratings['item'] == item),"rating"] += implicit_rating
    ratings.to_csv("./static/img/ratings.csv")
####################################################

def hello(request):
    context = {}
    context['content1'] = 'Hello World'
    return render(request, 'helloworld.html', context)


def get_location():
    url_loc = "https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyA24afb5VJ2UD1Y0sdfvJU2oouGaWzjnAE"
    r = req.post(url = url_loc, json = {"key":"value"})
    location = r.json()
    latitude = location['location']['lat']
    longitude = location['location']['lng']
    return str(latitude) + ',' + str(longitude)

@csrf_exempt
def goUpdate(request):
    item = str(request).split('?')[1][:-2]
    portfolio.item = item
    portfolio.user = recognize()
    expression = FEC()
    update_recommend(portfolio.user,portfolio.item,expression)
    return HttpResponse('Succcess Get Item Clicked')


emotions = ['Angry', 'Happy', 'Neutral', 'Confused']
search_url = 'https://www.googleapis.com/youtube/v3/search'
DEVELOPER_KEY = '' # provide API key
@csrf_exempt
def goData(request):
    global emotion_label
    global CFTable

    links = [
        {
            'Name': 'HAPPY Music - Good Morning Ukulele Music - The Best SUMMER Music',
            'URL': 'https://www.youtube.com/watch?v=NvZtkt9973A'
        },
        {
            'Name': 'Sad Piano Music (THIS WILL MAKE YOU CRY / Saddest Piano & Violin Ever!)',
            'URL': 'https://www.youtube.com/watch?v=aWIE0PX1uXk'
        },
        {
            'Name': 'Emotions Series - Anger | Most Epic Angry Dark Music Mix',
            'URL': 'https://www.youtube.com/watch?v=1qKS51qh4OY'
        },
        {
            'Name': 'Instrumental Music - Confused',
            'URL': 'https://www.youtube.com/watch?v=IDzP4O6ZDvs'
        }
    ]

    current_date = datetime.datetime.now()
    # to_search = np.random.choice(emotions)
    to_search = emotions[emotion_label]
    print(to_search)
    search_params = {
            'part': 'snippet',
            'q': to_search,
            'key': DEVELOPER_KEY,
            'maxResults': 50,
            'type': 'video',
    }

    print('user:',portfolio.user)

    id = CFTable.filter(CFTable['user'] == str(portfolio.user)).select('recommendations').collect()
    # id = CFTable.filter(CFTable['user'] == str(3)).select('recommendations').collect()
    print(id)
    item_id = []
    for row in range(len(id)):
        i = 0
        while True:
            try:
                item_id.append(id[row]['recommendations'][i][0])
                i+=1
            except:
                break

    with open("./static/img/items.json", "r") as f:
        items = json.load(f)

    target_url = []
    for url in items:
        if items[url] in item_id:
            print('title:', get_title(url))
            target_url.append({'Name': get_title(url), 'URL': 'https://www.youtube.com/watch?v=' + url})

    print(target_url)

    r = requests.get(search_url, params=search_params)
    try:
        results = r.json()['items']

        videos = []
        for search_result in results:
            videos.append({'Name': search_result['snippet']['title'], 'URL': 'https://www.youtube.com/watch?v=' + search_result['id']['videoId']})



        print("Video No: ", len(videos))
        if len(videos) != 0:
            links = np.random.choice(videos, 6-len(target_url))

        target_url.extend(links)
    except:
        pass

    return JsonResponse(list(target_url), safe=False)

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

        portfolio.user = recognize()
        portfolio.item = None



    print(time.time() - start)
    return render(request, 'recommend.html', context)

@csrf_exempt
def getRecommend(request):
    context = {}
    context['content1'] = 'Hello World'
    return render(request, 'getRecommend.html', context)
