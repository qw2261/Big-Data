from django.http import HttpResponse
from django.shortcuts import render
import pandas_gbq
from google.oauth2 import service_account

# Make sure you have installed pandas-gbq at first;
# You can use the other way to query BigQuery.
# please have a look at
# https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-nodejs
# To get your credential

credentials = service_account.Credentials.from_service_account_file('./hw4_tutorial/Homework0-2e77b0e16412.json')


def hello(request):
    context = {}
    context['content1'] = 'Hello World!'
    return render(request, 'helloworld.html', context)


def dashboard(request):
    pandas_gbq.context.credentials = credentials
    pandas_gbq.context.project = "homework0-253123"

    SQL = "SELECT SAFE.SUBSTR(time, 12, 5) AS time, SUM(ai) AS ai, SUM(data) AS data, SUM(good) AS good, SUM(movie) \
          AS movie, SUM(spark) AS spark FROM `homework0-253123.hw_3.cleaned`  \
          GROUP BY time ORDER BY time limit 8"

    df = pandas_gbq.read_gbq(SQL)

    data = {'data': []}


    for each in df.to_numpy():
        data['data'].append({'Time': each[0],
                             'count': {'ai': each[1],
                                       'data': each[2],
                                       'good': each[3],
                                       'movie': each[4],
                                       'spark': each[5]}})


    '''
        TODO: Finish the SQL to query the data, it should be limited to 8 rows. 
        Then process them to format below:
        Format of data:
        {'data': [{'Time': hour:min, 'count': {'ai': xxx, 'data': xxx, 'good': xxx, 'movie': xxx, 'spark': xxx}},
                  {'Time': hour:min, 'count': {'ai': xxx, 'data': xxx, 'good': xxx, 'movie': xxx, 'spark': xxx}},
                  ...
                  ]
        }
    '''

    return render(request, 'dashboard.html', data)


def connection(request):
    pandas_gbq.context.credentials = credentials
    pandas_gbq.context.project = "homework0-253123"
    SQL1 = 'SELECT * FROM `homework0-253123.hw4.nodes`'
    df1 = pandas_gbq.read_gbq(SQL1)

    SQL2 = 'SELECT * FROM `homework0-253123.hw4.links`'
    df2 = pandas_gbq.read_gbq(SQL2)

    data = {'n':[], 'e':[]}

    for each in df1.to_numpy():
        data['n'].append({'node': each[0]})

    for each in df2.to_numpy():
        data['e'].append({'source': each[0], 'target': each[1]})

    '''
        TODO: Finish the SQL to query the data. 
        Then process them to format below:
        Format of data:
        {'n': [{'node': 18233}, {'node': 18234}, ...],
         'e': [{'source': 0, 'target': 0},
                {'source': 0, 'target': 1},
                ...
                ]
        }
    '''
    return render(request, 'connection.html', data)
