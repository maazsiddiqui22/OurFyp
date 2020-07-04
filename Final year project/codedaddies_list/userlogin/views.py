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

def landing_page(request):
    return render(request,'landingpage.html')

def login(request):

    if request.POST:
        email= request.POST['email']
        password = request.POST['password']

        count=Userlogin.objects.filter(email=email, password=password).count()
         
       
        group = Userlogin.objects.filter(email=email, password=password)
        clusetr =0
        for i in group:
            clusetr = (( str(i).split(' ') )[0] )
        
        
        if count >0:
            det = detect(request)
            print("mood + GROUP = ",det+clusetr)
            
            strn = 'home/website2'
            y_light,y_dark,o_light,o_dark , size1,clr,size2 = getInterface(int(clusetr),det )
            file= open('result.txt','w')
            file.write(y_light+','+y_dark+','+o_light+','+o_dark+','+size1+','+clr+','+size2 )
            return redirect('/home/website2')


        else:
            messages.error(request,'Invalid email or password')

            return redirect('login')
   
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
       import datetime

       x = datetime.datetime.now()


       age =int(x.strftime("%Y"))-int(dob)
       if('y' in similarity or 'Y' in similarity):
           similarity='1'
           print((int(education),age,0,int(gender)))
           group = getCluster(int(education),age,1,int(gender))
       else:
           similarity ='0'
           print((int(education),age,0,int(gender)))
           group = getCluster(int(education),age,0,int(gender))

       obj = Userlogin(username=username,email=email,password=password,gender=gender,country=country,dob=dob, similarity=similarity, 
       domain=domain,education =education,group=group )
       obj.save()

       return redirect('login')


def logout(request):
    return "hello" 

def normalize(array,max,min):
    arr_to = []
    for element in array:
        arr_to.append((max-int(element))/(max-min) )
    return arr_to


def clusteringKMEAN():
    from sklearn import preprocessing
    from sklearn.cluster import KMeans
    import pickle
    print('clustering .. .. ..')
    data = pd.read_csv('data.csv')
    # max=data['Age'].max()
    # min=data['Age'].min()
    # array_age = normalize( data['Age'],max,min)
    # array_age = pd.DataFrame(array_age,columns=['Age'])
    
    data2 = data['Age'].to_numpy()
    # print(data2.reshape(-1,1))
    clustering = KMeans(n_clusters=4, random_state=0).fit(data2.reshape(-1,1))
    filename = 'finalized_model.pkl'
    pickle.dump(clustering, open(filename, 'wb'))
   
    

def getCluster(education,age,smilarity,gender):
    import pickle
    
    import os

    if os.path.isfile('finalized_model.pkl'):
        pass
    else:
        print('RE LEARNING KMEAN!!!!!!!!!!!!!!!!!!!')
        clusteringKMEAN()

    clustering = pickle.load(open('finalized_model.pkl', 'rb'))
    result = clustering.predict([[age]])
        
    return result[0]
 


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
    f = open('./userlogin/model.h5', 'r')
    myfile = File(f)
    model.load_weights('./userlogin/model.h5')
    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Sad", 2: "Angry", 3: "Happy", 4: "Sad", 5: "Sad", 6: "Happy"}
    # emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
   
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
   
    filename = open('./userlogin/haarcascade_frontalface_default.xml', 'r')
    myfile2 = File(filename)
    facecasc = cv2.CascadeClassifier('./userlogin/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    #cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    #ret,frame = cap.read() # return a single frame in variable `frame`
    
    for (x, y, w, h) in faces:
        
        try:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            break
        except:
            redirect('login')

    cv2.imwrite('c1.png',frame)
        
        
    #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

    #cap.release()
    #cv2.destroyAllWindows()
    
    # file= open('result.txt','w')
    

    # if str(emotion_dict[maxindex]) == 'Happy':
         
        
    #     file.write("0277bd,")
    #     file.write("ad1457,") 
    #     file.write("Happy,")
    # elif str(emotion_dict[maxindex]) == 'Neutral':
        
    #     file.write("4caf50,")
    #     file.write("ad1457,")
    #     file.write("Neutral,")
    #     #print('neutral')
    # elif str(emotion_dict[maxindex]) == 'Angry':
        
    #     file.write("b71c1c,")
    #     file.write("ad1457,") 
    #     file.write("Angry,")
    # elif str(emotion_dict[maxindex]) == 'Sad':
    #     file.write("fb8c00,")
    #     file.write("ad1457, ") 
    #     file.write("Sad,")
           
    
    # elif str(emotion_dict[maxindex]) == 'Surprised':
    #     file.write("b71c1c,") 
    #     file.write("ad1457, ") 
    #     file.write("Surprised,")
          
        
        
    
    return emotion_dict[maxindex]

def getInterface(cluster,mood):

    if cluster == 0:
        if mood == 'Sad':
            return '#ffeb99','#ffcc00','#ffc299','#ff8533','19','black','20'
        elif mood =='Angry':
            return '#99e6ff','#33ccff','#66ffb3','#00e673','19','black','20'
        else:
            return '#bf00ff', '#5f1782','#bf00ff', '#5f1782','19','white','20'
    elif cluster ==1:
        if mood == 'Sad':
            return '#ffe6b3','#ffc34d','#ffb366','#ff8c1a','18','black','19'
        elif mood =='Angry':
            return '#b3ecff','#4dd2ff','#b3ffe0','#4dffb8','18','black','19'
        else:
            return '#bf00ff', '#5f1782','#bf00ff', '#5f1782','18','white','19'
    elif cluster == 2:
        if mood == 'Sad':
            return '#fff5cc','#ffdb4d','#ffcc99','#ff9933','20','black','22'
        elif mood =='Angry':
            return '#ccebff','#66c2ff','#ccffdd','#66ff99','20','black','22'
        else:
            return '#bf00ff', '#5f1782','#bf00ff', '#5f1782','20','white','22'
    elif cluster ==  3:
        if mood == 'Sad':
            return '#ffd480','#ffb31a','#ffd699','#ffad33','17','black','17'
        elif mood =='Angry':
            return '#99ddff','#33bbff','#b3ffd9','#33ff99','17','black','17'
        else:
            return '#bf00ff', '#5f1782','##bf00ff', '#5f1782','17','white','17'


 


 

   

