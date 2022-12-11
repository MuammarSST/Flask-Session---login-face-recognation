from flask import Flask, render_template, redirect, request, session, Response, url_for, jsonify
# The Session instance is not used for direct access, you should always use flask.session
from flask_session import Session

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

import cv2
import mysql.connector
from PIL import Image
import numpy as np
import os
import time
from datetime import date



mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="aspire4730z",
    database="flask_db"
)
mycursor = mydb.cursor()
camera = cv2.VideoCapture(0) 

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result




@app.route("/ceklogin", methods=["POST", "GET"])
def ceklogin():
    if request.method == "POST":
        
        username = request.form.get("username")
        password = request.form.get("password")
        mycursor.execute("select * from user where username ='{}' and password='{}'".format(username,password))
        data =mycursor.fetchall()
        session["user_id"] = data[0]
        return redirect("/")

@app.route("/login")
def login():
    return render_template('login.html')
    

@app.route('/')
def home():
    if not session.get("user_id"):
        return redirect("/login")
    mycursor.execute("select * from user")
    data = mycursor.fetchall()
 
    return render_template('index.html', data=data)

@app.route('/admin')
def admin():
    return render_template('admin_login.html')

@app.route('/tambah_user')
def tambah_user():
    mycursor.execute("select ifnull(max(user_id) + 1, 100) from user")
    row = mycursor.fetchone()
    id_user = row[0]
    return render_template('tambah_user.html',id_user=int(id_user))

@app.route('/simpan_user', methods=['POST'])
def simpan_user():
    user_id = request.form.get('txt_id')
    username = request.form.get('txt_username')
 
    mycursor.execute("""INSERT INTO `user` (`user_id`, `username`) VALUES
                    ('{}', '{}')""".format(user_id, username))
    mydb.commit()
 
    return redirect(url_for('vfdataset_page', user_id=user_id))

@app.route('/vfdataset_page/<user_id>')
def vfdataset_page(user_id):
    return render_template('gendataset.html', user_id=user_id)
 
@app.route('/vidfeed_dataset/<user_id>')
def vidfeed_dataset(user_id):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_dataset(user_id):
    face_classifier = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    count_img = 0
    max_imgid=100
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
           
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+user_id+"-"+ str(count_img) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13  or int(count_img) == int(max_imgid) :
                break
                cap.release()
                cv2.destroyAllWindows()
 

@app.route('/train_classifier/<user_id>')
def train_classifier(user_id):
    dataset_dir = "dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split("-")[0])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.json")
 
    return redirect('/')



@app.route("/logout")
def logout():
	session["user_id"] = None
	return redirect("/")

face_cascade=cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
       #extracting frames
        ret, frame = self.video.read()
        frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
        interpolation=cv2.INTER_AREA)                    
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            recognizer = cv2.face.LBPHFaceRecognizer_create() 
            recognizer.read("classifier.json")
            id, pred = recognizer.predict(gray[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            if confidence > 70 :
                mycursor.execute("select * from user where user_id ='{}'".format(id))
                row = mycursor.fetchone()
                user_id = row[0]
                user_name = row[1]

                cv2.putText(frame, str('{}, {}%'.format(user_name,round(confidence,2))), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
               
            break        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/login_video_feed')
def login_video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/loadData', methods = ['GET', 'POST'])
def loadData():
    mycursor.execute("select * from user where user_id ='{}'".format(id))
    data = mycursor.fetchall()
 
    return jsonify(response = data)

@app.route("/periksalogin", methods=["POST", "GET"])
def periksalogin():
    if request.method == "GET":
        
        username = request.form.get("username")
        password = request.form.get("password")
        print(username)
        print(password)
        
        mycursor.execute("select * from user where username ='{}' and password='{}'".format(username,password))
        data =mycursor.fetchall()
        
        

if __name__ == "__main__":
	app.run(debug=True)
