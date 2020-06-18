from django.db import models


class Userlogin (models.Model):
    username= models.CharField('User name' , max_length=100)
    email= models.CharField('Email' , max_length=100)
    password= models.CharField('Password',max_length=20)

# Create your models here.




  