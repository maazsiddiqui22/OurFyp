from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files import File
from .models import *
from django.contrib import messages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def login(request):

    if request.POST:
        email= request.POST['email']
        password = request.POST['password']

        count=Userlogin.objects.filter(email=email, password=password).count()
        print(count)
        if count >0:
            detect(request)
            strn = 'home/website2'
            
            print('should not run this')
            return redirect(strn)

        else:
            messages.error(request,'Invalid email or password')

            return redirect('login.html')
   
    return render(request,'login.html')   

def signup(request):
    return render(request,'signup.html')

def register_user(request):
    if request.POST:
       username = request.POST['username']
       email = request.POST['email']
       password = request.POST['password']
       dob =request.POST['dob'].split('-')
       dob = dob[0]
       country= request.POST['country']
       gender =request.POST['gender']
       similarity = request.POST['similarity']
       domain = request.POST['domain']
       education = request.POST['education']
       if('y' in similarity or 'Y' in similarity):
           similarity='1'
       else:
            similarity ='0'
    #    if(gender=='Male'):
    #        group= relearnKnn(1,int(dob))
    #    else:
    #        group= relearnKnn(0,int(dob))

       print('sim,dom=',similarity, domain,education)

       obj = Userlogin(username=username,email=email,password=password,gender=gender,country=country,dob=dob, similarity=similarity, 
       domain=domain,education =education )
       obj.save()

       return redirect('login')


def logout(request):
    return "hello" 


def relearnKnn(a,b):
    
    dataset =pd.read_csv('dataForKnn.csv')

    le = preprocessing.LabelEncoder()
    genderEncoded=le.fit_transform( dataset['Gender'])
    genderEncoded = pd.DataFrame(genderEncoded)
    countryEncoded = le.fit_transform( dataset['Country'])
    countryEncoded =pd.DataFrame(countryEncoded)
    x =pd.concat( [genderEncoded,dataset['Year']],axis=1)
    y = pd.DataFrame(dataset['Label'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    result = knn.predict([[a,b]])
    return result[0]



def detect(request):
    # Create the model
    model = Sequential()
    time.sleep(2)

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    f = open('D:/django/codedaddies_list/codedaddies_list/userlogin/model.h5', 'r')
    myfile = File(f)
    model.load_weights('D:/django/codedaddies_list/codedaddies_list/userlogin/model.h5')
    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
   
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
   
    filename = open('D:/django/codedaddies_list/codedaddies_list/userlogin/haarcascade_frontalface_default.xml', 'r')
    myfile2 = File(filename)
    facecasc = cv2.CascadeClassifier('D:/django/codedaddies_list/codedaddies_list/userlogin/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    #cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    #ret,frame = cap.read() # return a single frame in variable `frame`
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        print(maxindex)
        print(emotion_dict[maxindex])
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite('D:/Final Year Project/codedaddies_list/userlogin/c1.png',frame)
        
        
    #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

    #cap.release()
    #cv2.destroyAllWindows()
    
    file= open('result.txt','w')
    

    if str(emotion_dict[maxindex]) == 'Happy':
         
        
        file.write("0277bd,")
        file.write("ad1457,") 
        file.write("Happy,")
    elif str(emotion_dict[maxindex]) == 'Neutral':
        
        file.write("4caf50,")
        file.write("ad1457,")
        file.write("Neutral,")
        #print('neutral')
    elif str(emotion_dict[maxindex]) == 'Angry':
        
        file.write("b71c1c,")
        file.write("ad1457,") 
        file.write("Angry,")
    elif str(emotion_dict[maxindex]) == 'Sad':
        file.write("fb8c00,")
        file.write("ad1457, ") 
        file.write("Sad,")
           
    
    elif str(emotion_dict[maxindex]) == 'Surprised':
        file.write("b71c1c,") 
        file.write("ad1457, ") 
        file.write("Surprised,")
          
        
        
    
    return emotion_dict[maxindex]

 


 

   

