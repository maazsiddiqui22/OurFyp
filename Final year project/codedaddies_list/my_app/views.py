import requests, webbrowser
from bs4 import BeautifulSoup
from django.shortcuts import render
from requests.compat import quote_plus
from . import models

BASE_CRAIGSLIST_URL = 'https://losangeles.craigslist.org/search/sss?query={}'
BASE_IMAGE_URL = 'https://images.craigslist.org/{}_300x300.jpg'

def website2(request):
    # file = open('result.txt' ,'r')
    # moodRead = file.readline().split(',')
    # bg_Color = moodRead[0]
    # mood = moodRead[2]
    # btnColor = moodRead[1]
    # print(btnColor)
    
    style={'light':'#ffe066', 'dark':'#ffcc00'}
    return render(request,'website2.html',style)

def home(request):
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    bg_Color = moodRead[0]
    mood = moodRead[2]
    btnColor = moodRead[1]
    print(btnColor)
    style={'bgColor': bg_Color, 'btnColor':btnColor}
    
    return render(request, 'base.html',style)



# def home(request,mood):
#     print('mood====>',mood)
#     if str(mood) == 'Happy':
#         style = {
#             'bgColor': '0277bd',
#             'mood':'Happy'
#         }
#     elif str(mood) == 'Neutral':
#         style = {
#             'bgColor': '4caf50',
#             'mood':'Neutral'
#         }
#     elif str(mood) == 'Angry':
#         style = {
#             'bgColor': 'b71c1c',
#             'mood':'Angry'
#         }
#     elif str(mood) == 'Sad':
#         style = {
#             'bgColor': 'fb8c00',
#             'mood':'Sad'
#         }
    
#     elif str(mood) == 'Surprised':
#         style = {
#             'bgColor': 'fb8c00',
#             'mood':'Surprised'
#         }
#     return render(request, 'base.html', style)


def new_search(request):
    search = request.POST.get('search')
    models.Search.objects.create(search=search)
    final_url = BASE_CRAIGSLIST_URL.format(quote_plus(search))
    print(final_url)
    response = requests.get(final_url)
    
    # print(response.text)
    soup = BeautifulSoup(response.text, 'html.parser')
    post_titles = soup.find_all('li', {'class': 'result-row'})
    print(post_titles[0])
    final_postings =[]

    for post in post_titles:
        post_title = post.find(class_='result-title').text
        post_url = post.find('a').get('href')
        post_price = post.find(class_='result-price').text
        
        if post.find(class_='result-image').get('data-ids'):
            post_image_id = post.find(class_='result-image').get('data-ids').split(',')[0].split(':')[1]
            post_image_url = BASE_IMAGE_URL.format(post_image_id)
            print(post_image_url)

        else:
            post_image_url = 'https://craigslist.org/images/peace.jpg'
        final_postings.append((post_title,post_url,post_price,post_image_url))
    
    file = open('result.txt' ,'r')
    moodRead = file.readline().split(',')
    bg_Color = moodRead[0]
    mood = moodRead[1]

    
    
    stuff_for_frontend = {
        'search': search,
        'final_postings': final_postings,
        'bgColor': bg_Color,
    }
    return render(request, 'my_app/new_search.html', stuff_for_frontend)