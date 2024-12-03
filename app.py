from flask import render_template, Flask, request, jsonify, redirect,session,url_for
from flask_cors import CORS, cross_origin
import os
from Encode_Decode_utils.utils import decodeImage
from predict import plantleaf
from soil import fruit_soil_recommendations, vegetable_soil_recommendations
import sqlite3

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

def connect_db():
    conn = sqlite3.connect('users.db')
    return conn
def create_table():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
     (id INTEGER PRIMARY KEY AUTOINCREMENT,
      name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    mobile VARCHAR(15) NOT NULL)''')
    conn.commit()
def insert_user(name, email, mobile, username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, email, mobile, username, password) 
        VALUES (?, ?, ?, ?, ?)
    ''', (name, email, mobile, username, password))
    conn.commit()
    conn.close()
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']

        insert_user(name, email, mobile, username, password)
        return redirect('/')
    return render_template('register.html')
# Function to authenticate user login
def authenticate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND password = ?
    ''', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['name'] = user[1]
            session['email'] = user[2]
            return redirect('/home')
        else:
            return render_template('login.html', message='Invalid username or password.')
    return render_template('login.html')
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route("/home", methods=['GET'])
@cross_origin()
def home():
    if request.method == "GET":
        return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   if request.method == "POST":
    image_file = request.files['file']
    print(image_file)
    #decodeImage(image, clApp.filename)
    classifier = plantleaf()  # creating a object of plantleaf class
    result = classifier.predictPlantImage(image_file)
    return result
   else:
       print('Loading Error')

@app.route('/soil')
def soil():
    fruits = fruit_soil_recommendations.keys()
    vegetables = vegetable_soil_recommendations.keys()
    return render_template('soil.html', fruits=fruits, vegetables=vegetables)

@app.route('/recommend_soil', methods=['POST'])
def recommend_soil():
    selected_crop = request.form['crop']
    selected_category = request.form['category']
    
    if selected_category == 'fruit':
        recommendation = fruit_soil_recommendations.get(selected_crop)
    elif selected_category == 'vegetable':
        recommendation = vegetable_soil_recommendations.get(selected_crop)
    else:
        return "Invalid category selected."
    
    if recommendation:
        soil_type = recommendation['soil_type']
        reason = recommendation['reason']
        return render_template('result.html', selected_crop=selected_crop, soil_type=soil_type, reason=reason)
    else:
        return "Soil recommendation not available for the selected crop."


if __name__ == "__main__":
    create_table()
    app.run(debug=True)