# Import libraries
import pandas as pd
from ultralytics import YOLO
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
import shutil


## Creating a tmp folder
try:
    os.mkdir('/tmp')
except:
    pass

def clr_pred():
    """
    for removing the static folder (for memoery saving in web)
    """
    dir = '/tmp/static'
    try:
        shutil.rmtree(dir)
        print("Cleared Previous Predictions")
    except:
        pass

def clr_old_upload():
    """
    for removing the old images uploaded file (for memoery saving in web)
    """
    dir = '/tmp/UPLOAD_FOLDER'
    try:
      shutil.rmtree(dir)
      print("Cleared UPLOAD_FOLDER")
    except:
        pass


#making dir to save the generated csv file
def mk_csv_folder():
    try:
        os.mkdir("/tmp/static/CSV_File")
        print("made csv_folder")
    except:
        pass

#Creating UPLOAD_FOLDER dir to save the Uploaded file
def mk_uploaded_folder():
    try:
        os.mkdir("/tmp/UPLOAD_FOLDER")
        print("made UPLOAD_FOLDER")
    except:
        pass

#Creating static dir to save the generated file
def mk_pred_folder():
    try:
        os.mkdir("/tmp/static")
        print("made static folder")
    except:
        pass


# creating the model  instance
model = YOLO('best.pt')


def Use_yolo(img_path):
    """
    :param img_path:
    :return: image with bbox , csv file
    """
    all_data =[]

    results = model(img_path, conf=0.1, verbose=False)
    model.predict(img_path, save=True, conf=0.2, show_labels=True,
                  project='/tmp/static', name="Image_Prediction")
    # Extract bounding boxes, confidence scores, and class labels
    boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes in xyxy format
    classes = results[0].boxes.cls.tolist()  # Class indices
    confidences = results[0].boxes.conf.tolist()  # Confidence scores
    names = results[0].names  # Class names dictionary

    if not boxes:
        # If no detections, add NEG as the class
        all_data.append({
            'Class': 'No Object Found',  # Default value (no detection)
            'x_min': 'Nan',  # Default value (no detection)
            'y_min': 'Nan',  # Default value (no detection)
            'x_max': 'Nan',  # Default value (no detection)
            'y_max': 'Nan'  # Default value (no detection)
        })
    else:
        # Iterate through the results for this image
        for box, cls, conf in zip(boxes, classes, confidences):
            x_min, y_min, x_max, y_max = box
            detected_class = names[int(cls)]  # Get the class name from the names dictionary

            # Add the result to the all_data list
            all_data.append({
                'Class': detected_class,
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'Confidence/Probability Score': conf
            })

    sub = pd.DataFrame(all_data)

    sub.to_csv("/tmp/static/CSV_File/WBC_File.csv", index=False)


app = Flask(__name__, static_folder="/tmp")
app.config['UPLOAD_FOLDER'] = '/tmp/UPLOAD_FOLDER'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def home_click():
    return render_template('index.html')

@app.route('/document')
def document_click():
    df = pd.read_csv("Documents/results.csv")  # Reading CSV File
    # Convert DataFrame to a list of dictionaries
    data = df.to_dict(orient='records')

    return render_template('document.html', data=data, columns=df.columns)


@app.route('/wbc_info')
def wbc_info_click():
    return render_template('wbc_info.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            ## clearing old files and folders and creating Folders for saving file
            clr_old_upload()
            clr_pred()
            mk_pred_folder()
            mk_uploaded_folder()
            mk_csv_folder()
            show_csv_heading = False  ## This is set so that co-ordinates table heading will only whow when it is true
            f = request.files['fileInput'] ## geting path of input file

            f.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename) )) ## saving the input image file in UPLOAD_FOLDER
            imageList = os.listdir("/tmp/UPLOAD_FOLDER")  # geting listv of image files in UPLOAD_FOLDER

            for image in imageList:
                ### Applying yolo model for object detection on uploaded files
                Use_yolo("/tmp/UPLOAD_FOLDER/"+image)

            pred_image_list = os.listdir("/tmp/static/Image_Prediction") ## geting the file path of generated image having object detection

            df = pd.read_csv("/tmp/static/CSV_File/WBC_File.csv")  # Reading CSV File
            # Convert DataFrame to a list of dictionaries
            data = df.to_dict(orient='records')
            show_csv_heading = True
            return render_template("index.html", pred_image_list= pred_image_list, data=data, columns=df.columns, show_csv_heading=show_csv_heading)

    except:
        return render_template("error.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)
