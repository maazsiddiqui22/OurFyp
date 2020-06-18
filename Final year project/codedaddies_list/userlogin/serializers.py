from rest_framework import serializers
from .models import Userlogin 

class UserLoginSerializers(serializers.ModelSerialzers):
    class  Meta:
        model=Userlogin 
        fields='__all__'