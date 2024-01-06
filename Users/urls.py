from django.urls import path
from .import views
urlpatterns=[
    path('',views.home,name='homepage'),
    path('index',views.index,name='index'),
    path('crop_predict',views.crop_predict,name='croppredict')
]