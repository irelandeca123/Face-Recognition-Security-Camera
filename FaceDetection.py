#https://smallguysit.com/index.php/2017/03/13/python-tkinter-password-entry-box/
import tkinter
import cv2 
import os
import numpy as np
from PIL import Image
import pickle
from google.cloud import storage
import re
import pyrebase
import json
import objectpath
import time
from imutils.video import VideoStream
import imutils
import datetime




main = tkinter.Tk()
main.title('Authentication Box')
main.geometry('225x150')

def clear_widget(event):

    # will clear out any entry boxes defined below when the user shifts
    # focus to the widgets defined below
    if username_box == main.focus_get() and username_box.get() == 'Enter Username':
        username_box.delete(0, tkinter.END)
    elif password_box == password_box.focus_get() and password_box.get() == '     ':
        password_box.delete(0, tkinter.END)

def repopulate_defaults(event):

    # will repopulate the default text previously inside the entry boxes defined below if
    # the user does not put anything in while focused and changes focus to another widget
    if username_box != main.focus_get() and username_box.get() == '':
        username_box.insert(0, 'Enter Username')
    elif password_box != main.focus_get() and password_box.get() == '':
        password_box.insert(0, '     ')

def login(*event):
    global email, password
    # Able to be called from a key binding or a button click because of the '*event'
    print (('Username: ') + username_box.get())
    print (('Password: ') + password_box.get())
    

    email = username_box.get() 
    password = password_box.get() 
    
    
    main.destroy()
    
    
    
# defines a grid 50 x 50 cells in the main window
rows = 0
while rows < 10:
    main.rowconfigure(rows, weight=1)
    main.columnconfigure(rows, weight=1)
    rows += 1


# adds username entry widget and defines its properties
username_box = tkinter.Entry(main)
username_box.insert(0, 'Enter Username')
username_box.bind("<FocusIn>", clear_widget)
username_box.bind('<FocusOut>', repopulate_defaults)
username_box.grid(row=1, column=5, sticky='NS')


# adds password entry widget and defines its properties
password_box = tkinter.Entry(main, show='*')
password_box.insert(0, '     ')
password_box.bind("<FocusIn>", clear_widget)
password_box.bind('<FocusOut>', repopulate_defaults)
password_box.bind('<Return>', login)
password_box.grid(row=2, column=5, sticky='NS')


# adds login button and defines its properties
login_btn = tkinter.Button(main, text='Login', command=login)
login_btn.bind('<Return>', login)
login_btn.grid(row=5, column=5, sticky='NESW')


main.mainloop()




os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="camera-detection-73a01-firebase-adminsdk-e31px-f129000ad6.json"

# Enable Storage
client = storage.Client()

# Reference an existing bucket.
bucket = client.get_bucket('camera-detection-73a01.appspot.com')




config = {
    "apiKey": "AIzaSyBpRsAequKTjY25_ew-RutT31eE4COHU9E",
    "authDomain": "camera-detection-73a01.firebaseapp.com",
    "databaseURL": "https://camera-detection-73a01.firebaseio.com",
    "projectId": "camera-detection-73a01",
    "storageBucket": "camera-detection-73a01.appspot.com",
    "messagingSenderId": "56675147413"
}

firebase = pyrebase.initialize_app(config)

auth = firebase.auth()

#email = input ('Please enter your email\n')
#password = input ('Please enter your password\n')


print ("username: " + email,"password: " + password)


#user = auth.create_user_with_email_and_password(email,password)
user = auth.sign_in_with_email_and_password(email,password)
#user = auth.sign_in_with_email_and_password(email,password)
users = (auth.get_account_info(user['idToken']))

json.dump(users, open("users.txt",'w'))

tree_obj = objectpath.Tree(users)

displayName = tuple(tree_obj.execute('$..displayName'))

displayName = (displayName[1])

prefix='images/' + displayName + '/' 
folder = {}
char_list = ['/images/' + displayName]
list = {}
blobs = bucket.list_blobs(prefix=prefix)
dl_dir = 'images/' + displayName

iterator = bucket.list_blobs(prefix=prefix, delimiter='/')
prefixes = set()
for page in iterator.pages:
    prefixes.update(page.prefixes)
print (prefixes)
print (type(prefixes))
test = ', '.join(prefixes)
print (test)



def uploading_files():
    # Enable Storage
    client = storage.Client()
    
    # Reference an existing bucket.
    bucket = client.get_bucket('camera-detection-73a01.appspot.com')
    print (recognized)
  
#    if (recognized == "recognized"):
#        #Upload a local file to a new file to be created in your bucket.
#        zebraBlob = bucket.blob(prefix + name + "/"  + filename)
#        zebraBlob.upload_from_filename(filename=filename)
#    else:
#        recognized == ""
#        zebraBlob = bucket.blob(prefix + "unknown" + "/"  + filename)
#        zebraBlob.upload_from_filename(filename=filename)
    
    if 'recognized' in recognized :
       #Upload a local file to a new file to be created in your bucket.
        zebraBlob = bucket.blob(prefix + name + "/"  + filename)
        zebraBlob.upload_from_filename(filename=filename)
        
    if 'unrecognized' in recognized :
        zebraBlob = bucket.blob(prefix + "unknown" + "/"  + filename)
        zebraBlob.upload_from_filename(filename=filename)
         
        


def downloading_files():
    for blob in blobs:
        list = blob.name
            
        if list.endswith('jpg') or list.endswith('png'):
            list = re.sub("|".join(char_list), "", list)
            print(list)
            giraffeBlob = bucket.blob(list)
  
            with open(list, 'wb') as file_obj:
                giraffeBlob.download_to_file(file_obj)
     
                
                
def checking_folder():
    for folder in prefixes:
        if os.path.isdir(folder):    
            print ("The Folder Already Exists")
           
        else:
            print("The Folder Doesnt Exist")
            os.mkdir(folder)






def training_faces():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")
    
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    
    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []
    
    for root, dirs, files in os.walk(image_dir):
    	for file in files:
    		if file.endswith("png") or file.endswith("jpg"):
    			path = os.path.join(root, file)
    			label = os.path.basename(root).replace(" ", "-").lower()
    			#print(label, path)
    			if not label in label_ids:
    				label_ids[label] = current_id
    				current_id += 1
    			id_ = label_ids[label]
    			#print(label_ids)
    			#y_labels.append(label) # some number
    			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
    			pil_image = Image.open(path).convert("L") # grayscale
    			size = (550, 550)
    			final_image = pil_image.resize(size, Image.ANTIALIAS)
    			image_array = np.array(final_image, "uint8")
    			#print(image_array)
    			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)
    
    			for (x,y,w,h) in faces:
    				roi = image_array[y:y+h, x:x+w]
    				x_train.append(roi)
    				y_labels.append(id_)
    
    with open("pickles/face-labels.pickle", 'wb') as f:
    	pickle.dump(label_ids, f)
    
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizers/face-trainner.yml")
    


#def record_video():
#    global filename
#    # The duration in seconds of the video captured
#    capture_duration = 4
#    
#    cap = cv2.VideoCapture(0)
#    filename = ( datetime.datetime.now().strftime("%A%d%B%Y%I%M%S%p")+".avi")
#    print (filename)
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter(filename,fourcc, 20.0, (640,480))
#    
#    start_time = time.time()
#    while( int(time.time() - start_time) < capture_duration ):
#        ret, frame = cap.read()
#        if ret==True:
#            #frame = cv2.flip(frame,0)
#            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
#    		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#            out.write(frame)
#            #cv2.imshow('frame',frame)
#        else:
#            break
#    
#    cap.release()
#    out.release()
#    cv2.destroyAllWindows()
#    uploading_files()
#    main()



def face_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
   # eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
  #  smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
    global name
    #global unknown
    global recognized
    global filename
    
    recognized = [""]

    
    capture_duration = 10
    filename = ( datetime.datetime.now().strftime("%A%d%B%Y%I%M%S%p")+".avi")
    print (filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename,fourcc, 10, (640,480))
#    start_time = time.time()
    
    
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./recognizers/face-trainner.yml")
    
    labels = {"person_name": 1}
    with open("pickles/face-labels.pickle", 'rb') as f:
    	og_labels = pickle.load(f)
    	labels = {v:k for k,v in og_labels.items()}
    
 
    while(True):
        start_time = time.time()
        while(int(time.time() - start_time) < capture_duration):

        # Capture frame-by-frame
            ret, frame = cap.read()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
            	#print(x,y,w,h)
            	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            	roi_color = frame[y:y+h, x:x+w]
            	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            	(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                
        #        	t0 = time.time()   
            	out.write(frame)
            	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
            	id_, conf = recognizer.predict(roi_gray)
            	if conf>=3 and conf <= 85:
            		#print(5: #id_)
            		#print(labels[id_])
            		#print(conf)
            		font = cv2.FONT_HERSHEY_SIMPLEX
            		name = labels[id_]
            		color = (255, 255, 255)
            		stroke = 2
            		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            		if 'recognized' in recognized :
            			break
            		else:
            			recognized = recognized + ["recognized"]
                    
                    
            		#out.write(frame)
            	else:
            		font = cv2.FONT_HERSHEY_SIMPLEX                
            		name = labels[id_]            
            		color = (255, 255, 255)             
            		stroke = 2                
            		cv2.putText(frame, "unrecognized", (x,y), font, 1, color, stroke, cv2.LINE_AA)
            		unrecognized = "unrecognized.png"
            		cv2.imwrite(unrecognized,roi_color)   
            		if 'unrecognized' in recognized :
            			break
            		else:
            			recognized = recognized + ["unrecognized"]  
            		                                 

        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            	#subitems = smile_cascade.detectMultiScale(roi_gray)
            	#for (ex,ey,ew,eh) in subitems:
            	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # Display the resulting frame
            #cv2.imshow('frame',frame)
            #record_video()  
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        break
        #uploading_files()
        #main()

    
  
def motion_detection():
    vs = VideoStream(src=0).start()
    time.sleep(4.0)
    
    # initialize the first frame in the video stream
    firstFrame = None
    min_area = 500
    # loop over the frames of the video
    while True:
    	# grab the current frame and initialize the occupied/unoccupied
    	# text
    	frame = vs.read()
    	#frame = frame if args.get("video", None) is None else frame[1]
    	frame = frame
    	text = "Unoccupied"
    
    	# if the frame could not be grabbed, then we have reached the end
    	# of the video
    	if frame is None:
    		break
    
    	# resize the frame, convert it to grayscale, and blur it
    	frame = imutils.resize(frame, width=500)
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    	# if the first frame is None, initialize it
    	if firstFrame is None:
    		firstFrame = gray
    		continue
    
    	# compute the absolute difference between the current frame and
    	# first frame
    	frameDelta = cv2.absdiff(firstFrame, gray)
    	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    	# dilate the thresholded image to fill in holes, then find contours
    	# on thresholded image
    	thresh = cv2.dilate(thresh, None, iterations=2)
    	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
    	cnts = imutils.grab_contours(cnts)
    
    	# loop over the contours
    	for c in cnts:
    		# if the contour is too small, ignore it
    		if cv2.contourArea(c) < min_area:
    			continue
    
    		# compute the bounding box for the contour, draw it on the frame,
    		# and update the text
    		(x, y, w, h) = cv2.boundingRect(c)
    		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    		text = "Occupied"
    
    	# draw the text and timestamp on the frame
    	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
    		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    	
    
    	# show the frame and record if the user presses a key
    	cv2.imshow("Security Feed", frame)
    	#cv2.imshow("Thresh", thresh)
    	#cv2.imshow("Frame Delta", frameDelta)
    	key = cv2.waitKey(1) & 0xFF
    
    	# if the `q` key is pressed, break from the lop
    	if key == ord("q"):
    		break
    	if text == ("Occupied"): 
    		break
    # cleanup the camera and close any open windows
    
    vs.stream.release()
    vs.stop()
    cv2.destroyAllWindows()
    #face_detection()

def main():         
  #checking_folder()
  #downloading_files()
  training_faces()  
  motion_detection() 
  face_detection()
  uploading_files()
  
if __name__ == '__main__':
    main()
    
