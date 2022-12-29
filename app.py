import os
import pickle
import pandas as pd
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename

classifier = pickle.load(open('./models/gradientboosting_model.sav','rb'))
CSV_COLUMNS = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']

UPLOAD_FOLDER = './uploaded_files'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__, template_folder='./templates',static_url_path='',static_folder='./static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            #return redirect(request.url)
            return jsonify({"response":0})
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            #flash('No selected file')
            #return redirect(request.url)
            return jsonify({"response":0})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = predic(f'./uploaded_files/{filename}')
            print(type(result))
            #return redirect(url_for('download_file', name=filename))
            return send_file(result,download_name='prediction.csv',mimetype='text/csv')#jsonify({"response":1})

def predic(csv):
    if csv != None:
        df = pd.read_csv(csv)
        result = df.copy()
        sc = StandardScaler()
        if all(item in df.columns for item in CSV_COLUMNS):
            df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
            df = pd.get_dummies(df,columns=['Geography','Gender'])
            #data = sc.fit_transform(df)
            y_pred = classifier.predict(df)
            result = result.assign(Exit=y_pred)
            buffer = BytesIO() 
            result.to_csv(buffer,index=False)
            buffer.seek(0)
            return buffer
        else:
            return "columnas incorrectas"
    else:
        return "no hay archivo/vacio"
        
    

if __name__ == '__main__':
   app.run(debug=True)