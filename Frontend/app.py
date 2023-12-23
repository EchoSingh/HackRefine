from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = '.img//'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('dummy.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == 'ds.jpg':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        result = ml_prediction(filename)  # Perform ML prediction when the file is uploaded
        return redirect(url_for('result', result=result))

@app.route('/result/<result>')
def result(result):
    return render_template('result.html', result=result)

def ml_prediction(image_path):
    return "Example Prediction"

if __name__ == '__main__':
    app.run(debug=True)
