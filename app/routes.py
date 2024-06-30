from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app import app
from generate import generate_images

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Call function to generate images using your model
            output_image_path = generate_images(filepath,output_image_size=(128,128))

            return render_template('result.html', output_image=output_image_path)

    # If GET request or invalid file, redirect to index
    return redirect(url_for('index'))
