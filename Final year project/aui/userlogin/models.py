from django.db import models


class Userlogin (models.Model):
    username= models.CharField('User name' , max_length=100)
    country= models.CharField('Country' ,default='Pakistan', max_length=100)
    dob= models.CharField('DOB' ,default='1997', max_length=100)
    email= models.CharField('Email' , max_length=100)
    password= models.CharField('Password',max_length=20)
    gender = models.CharField('Gender',default='1',max_length=9)
    group = models.CharField('Group',default=0,max_length=2)
    education = models.CharField('Education' ,default='0', max_length=10)
    similarity = models.CharField('Similarity' ,default='0', max_length=10)
    domain = models.CharField('Domain' ,default='0', max_length=10)
    def __str__(self):
        return self.group + " " + self.email
# Create your models here.




