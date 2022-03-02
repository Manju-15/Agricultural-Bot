
from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/',methods=['GET'])
def hello():
    return render_template('backpicchange.html')
@app.route('/team',methods=['GET'])
def team():
    return render_template('team.html')
@app.route('/index1',methods=['GET'])
def index1():
    return render_template('index1.html')
@app.route('/About',methods=['GET'])
def About():
    return render_template('About.html')
@app.route('/services',methods=['GET'])
def services():
    return render_template('services.html')
@app.route('/search',methods=['GET'])
def search():
    return render_template('search.html')
@app.route('/manju',methods=['GET'])
def manju():
    return render_template('manju.html')
@app.route('/padmaja',methods=['GET'])
def padmaja():
    return render_template('padmaja.html')
@app.route('/sinegalatha',methods=['GET'])
def sinegalatha():
    return render_template('sinegalatha.html')
@app.route('/saaru',methods=['GET'])
def saaru():
    return render_template('saaru.html')
@app.route('/backpicchange',methods=['GET'])
def backpicchange():
    return render_template('backpicchange.html')

if __name__ == '__main__':
    app.run(debug=True)