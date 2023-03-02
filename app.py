
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf

class_names = ['Covid 19', 'NORMAL', 'Pnuemonia', 'Tubercolosis']
soln = ['Coronaviruses are a family of viruses that can cause illnesses such as the common cold, severe acute respiratory syndrome (SARS) and Middle East respiratory syndrome (MERS). In 2019, a new coronavirus was identified as the cause of a disease outbreak that originated in China.\n\nThe virus is known as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The disease it causes is called coronavirus disease 2019 (COVID-19). In March 2020, the World Health Organization (WHO) declared the COVID-19 outbreak a pandemic.','','Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.','Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air via coughs and sneezes.']
model = tf.keras.models.load_model("model/model.h5")

app = Flask(__name__)
app.secret_key="t34s"
app.config['UPLOAD_FOLDER'] = os.path.join('static','uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join("static/uploads/",filename))
			image = tf.keras.utils.load_img('static/uploads/'+filename, target_size=(224, 224))
			input_arr = tf.keras.utils.img_to_array(image)
			input_arr = np.array([input_arr])
			input_arr = tf.image.resize(input_arr,(224,224))
			predictions = model.predict(input_arr,verbose=0)
			score = tf.nn.softmax(predictions[0])
			predicted_class = class_names[np.argmax(predictions)]
			label, acc = predicted_class, float(100*np.max(score))
			flash(label)
			flash("{:.2f}".format(acc))
			flash(soln[np.argmax(predictions)])
			flash(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect('/')


if __name__ == "__main__":
    app.run()