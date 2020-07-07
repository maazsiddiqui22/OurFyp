import requests, webbrowser
from bs4 import BeautifulSoup
from django.shortcuts import render,redirect

def website2a(request):
    print('OPTION IS')
    option = request.POST['option1']
    print(option)
    print(type(option))
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    if option == '1':
        light =moodRead[2]
        dark = moodRead[3]
        
    else:
        light =moodRead[0]
        dark = moodRead[1]
    
    style={'light':light, 'dark':dark }
    return render(request,'website2.html',style)

def changeurl(request):
    value = request.POST['bttn2']
    print(request)
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    if value == moodRead[0]:
        style={'light':moodRead[2], 'dark':moodRead[3] , 'fontsize' : moodRead[4] , 'fontcolor' : moodRead[5], 'butnsize' : moodRead[6]}
    else:
        style={'light':moodRead[0], 'dark':moodRead[1], 'fontsize' : moodRead[4] , 'fontcolor' : moodRead[5], 'butnsize' : moodRead[6]}

    return render(request,'website2.html',style)

def website2(request):
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    print(moodRead)
    
    style={'light':moodRead[0], 'dark':moodRead[1],  'fontsize' : moodRead[4] , 'fontcolor' : moodRead[5], 'butnsize' : moodRead[6]}
    return render(request,'website2.html',style)

def getColor(num):
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    style={'light':moodRead[num], 'dark':moodRead[num+1]}
    return style