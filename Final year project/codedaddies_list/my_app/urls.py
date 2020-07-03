from django.urls import path
from . import views


urlpatterns =[
    path('/',views.home,name='home'),
    # path('/<mood>/',views.home,name='home'),
    path('/website2',views.website2,name='website2'),
    path('/changeurl' ,views.changeurl, name='changeurl'),
    path('/option',views.option,name='option'),
    path('/website2a',views.website2a,name='website2a'),
    path('/new_search/',views.new_search,name='new_search'),
    # path('<mood>/new_search/',views.new_search,name='new_search'),
]