from django.urls import path
from . import views


urlpatterns =[
    # path('/<mood>/',views.home,name='home'),
    path('/website2',views.website2,name='website2'),
    path('/changeurl' ,views.changeurl, name='changeurl'),
    path('/website2a',views.website2a,name='website2a')
]