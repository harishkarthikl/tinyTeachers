from flask import Flask, render_template, url_for, request, redirect, flash, Response, session
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_moment import Moment
import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

app = Flask(__name__)
app.config['SECRET_KEY'] = "sivy"

bootstrap = Bootstrap(app)
moment = Moment(app)

camera = cv2.VideoCapture(0)

#### START MNIST ML backend ####

def _zoom(im):
    #Function to zoom in or out
    choice = np.random.choice(['in', 'out'])
    if choice == 'in':
        im_new = ImageOps.crop(im, 0.85) #Crop 15% all around (zoom-in)
    else:
        width, height = im.size
        new_width = width + 45 #45 (or 15%) pixel padding to width
        new_height = height + 45 #45 (or 15%) pixel pading to height
        im_new = Image.new(im.mode, (new_width, new_height))
        im_new.paste(im, (100, 100)) #create new zoomed out image
    return im_new

def _flip(im):
    #Function to flip image horizontally or vertically
    choice = np.random.choice(['vertical', 'horizontal'])
    if choice == 'horizontal':
        im_new = im.transpose(Image.FLIP_LEFT_RIGHT) #flip image horizontally
    else:
        im_new = im.transpose(Image.FLIP_TOP_BOTTOM) #flip image horizontally
    return im_new

def _bright(im):
    #Function to increase or decrease brightness
    enh = ImageEnhance.Brightness(im)
    level = np.random.randint(7,14)/10
    im_new = enh.enhance(level) #Increase/Decrease brightness
    return im_new

def _shear(im):
    #Function to add shear
    width, height = im.size
    level = np.random.randint(-2,3)/10
    xshift = abs(level) * width
    new_width = width + int(round(xshift))
    img = im.transform((new_width, height), Image.AFFINE, (1, level, -xshift if level > 0 else 0, 0, 1, 0), Image.BICUBIC)
    return img

def _rotate(im):
    #Function to rotate image
    rot = np.random.randint(-30,31) #Select1an image rotation angle by random
    im_new = im.rotate(rot) #Random rotate
    return im_new

def _shift(im):
    #Function to shift image laterally or vertically
    arr = np.asarray(im)
    choice = np.random.choice(['l-r', 'u-d'])
    pixels = np.random.randint(-30, 31)
    if choice == 'l-r':
        arr = np.roll(arr, pixels, axis=1)
    else:
        arr = np.roll(arr, pixels, axis=0)
    im_new = Image.fromarray(arr)
    return im_new

def img_aug(image, obj_class):
    #Function to use all above functions to create 20 augmented images
    enh = ImageEnhance.Brightness(image)  # image enhancement to make it easy for our model to interpret
    im = enh.enhance(2.2)
    im = im.filter(ImageFilter.MinFilter(7))
    im = ImageOps.invert(im)  # invert black and white to match MNIST dataset
    filepath = os.path.join(os.getcwd(),'Images', obj_class, 'base.jpg')
    im.save(filepath)  # save native file with basic preprocessing

    for i in range(20):
        choice = np.random.choice(['zoom', 'flip'])
        if choice == 'zoom':
            img = _zoom(im)
        else:
            img = _flip(im)

        choice = np.random.choice(['bright', 'shear'])
        if choice == 'bright':
            img = _bright(img)
        else:
            img = _shear(img)

        choice = np.random.choice(['rotate', 'shift'])
        if choice == 'rotate':
            img = _rotate(img)
        else:
            img = _shift(img)
        filename = str(i) + '.jpg'
        filepath = os.path.join(os.getcwd(),'Images', obj_class, filename)
        img.save(filepath)  # save native file with basic preprocessing

class mnist_model(object):
    def __init__(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.layers import Dense

        #Check GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        physical_device = tf.config.experimental.list_physical_devices('GPU')
        print("# of GPU's available: ", len(physical_device))
        tf.config.experimental.set_memory_growth(physical_device[0], True)

        #Load base model trained on MNIST dataset
        base_model = keras.models.load_model('models/CNN-MNIST')
        for i in range(8):
            base_model.layers[i].trainable = False

        #Add layers to base model
        x = base_model.layers[7].output
        x = Dense(3, activation='softmax')(x)
        self.model = keras.Model(inputs=base_model.input, outputs=x)

    def train(self):
        from tensorflow import keras
        from tensorflow.keras.optimizers import Adam

        # Compile the model
        optimizer = Adam(1e-3)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        #Create augmented images
        A = Image.open('Images/ClassA/base.jpg').convert('L')
        img_aug(A, 'ClassA')
        B = Image.open('Images/ClassB/base.jpg').convert('L')
        img_aug(B, 'ClassB')
        C = Image.open('Images/ClassC/base.jpg').convert('L')
        img_aug(C, 'ClassC')

        #Add augmented images to data generator
        gen = keras.preprocessing.image.ImageDataGenerator()
        train_gen = gen.flow_from_directory('Images/',
                                            target_size=(28, 28),
                                            color_mode="grayscale",
                                            classes=None,
                                            class_mode="categorical",
                                            batch_size=13,
                                            shuffle=False,
                                            seed=99,
                                            save_to_dir=None,
                                            save_prefix="",
                                            save_format="jpg",
                                            follow_links=False,
                                            subset=None,
                                            interpolation="nearest")

        #Train model for 15 epochs
        self.model.fit_generator(train_gen, epochs=15)

    def predict(self, arr):
        pred = self.model.predict(arr)
        return pred

def test_img_process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = np.asarray(gray)
    img = Image.fromarray(a)
    enh = ImageEnhance.Brightness(img) #image enhancement to make it easy for our model to interpret
    im = enh.enhance(2.2)
    im = im.filter(ImageFilter.MinFilter(7))
    im = ImageOps.invert(im)
    im = im.resize((28,28))
    arr = np.asarray(im).reshape(1,28,28,1)
    return arr

#Instantiate the mnist_model class
see_model = mnist_model()

#### END MNIST ML backend ####

#### START Video ####

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read() #read frame
        #frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (170, 90), (470, 390), (0, 0, 255), 4)  #add rectangle
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def get_predict_frame(self):
        success, frame = self.video.read() #read frame
        #frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (170, 90), (470, 390), (0, 0, 255), 4)  #add rectangle

        arr = test_img_process(frame)
        pred = see_model.predict(arr)

        if np.argmax(pred) == 0:
            label = 'Class A  -  ' + str(np.round((np.max(pred[0]) * 100), 2))
        elif np.argmax(pred) == 1:
            label = 'Class B  -  ' + str(np.round((np.max(pred[0]) * 100), 2))
        else:
            label = 'Class C  -  ' + str(np.round((np.max(pred[0]) * 100), 2))

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (40, 40)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        # Using cv2.putText() method
        frame = cv2.putText(frame, label, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def get_frame_faces(self):
        success, frame = self.video.read()  # read frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Haar cascade needs monochrome images to recognize faces
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces_imgs = face_classifier.detectMultiScale(gray, 1.3, 5) # Identify all faces in various scales

        for (x,y,w,h) in faces_imgs:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)  # add rectangle
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

def gen_frames(camera):
    # for see page
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def gen_frames_faces(camera):
    # for faces page
    while True:
        frame = camera.get_frame_faces()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def gen_predict_frames(camera):
    while True:
        frame = camera.get_predict_frame()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def capture_frame(camera, folder):
    frame = camera.get_frame()
    arr = np.frombuffer(frame, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    base_path = os.getcwd()
    save_path = os.path.join(base_path, 'Images', folder, 'base.jpg')
    cv2.imwrite(save_path, image[90:390,170:470])
    return None


#### END Video ####


#### START Index Page ####

@app.route('/')
def index():
    return render_template('index.html')

#### END Index Page ####



#### START Faces Page ####

@app.route('/faces', methods=['GET', 'POST'])
def faces():
    return render_template('faces.html')

@app.route('/video_faces')
def video_faces():
    return Response(gen_frames_faces(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

#### END Faces Page ####



#### START See Page ####
@app.route('/see')
def see():
    return render_template('see.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    pred_class = str(request.args.get('pred_class'))
    capture_frame(VideoCamera(), pred_class)
    flash("Picture snapped!")
    return redirect('/see')

@app.route('/see_train')
def see_train():
    see_model.train()
    return redirect('/see')

@app.route('/see_predict')
def see_predict():
    return render_template('see_prediction.html')

@app.route('/prediction_video_feed')
def prediction_video_feed():
    return Response(gen_predict_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

#### END See Page ####



#### START Hear Page ####

@app.route('/hear', methods=['GET', 'POST'])
def hear():
    return render_template('hear.html')

#### END Hear Page ####



#### START About Page ####

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

#### END About Page ####



#### START Contact Page ####

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        first_name = request.form['fname']
        last_name = request.form['lname']
        return f"first name is {first_name} and last name is {last_name}"
        #return render_template('contact.html')
    else:
        return render_template('contact.html')

#### END Contact Page ####

if __name__ == '__main__':
    app.run(debug=True)