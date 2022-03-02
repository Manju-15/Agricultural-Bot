from flask import Flask,render_template,request
import pickle
import json
import numpy as np
import nltk
import torch
import torch.nn as nn
with open('C:\\Users\\anand\\PycharmProjects\\NAAC\\2-1.json', 'r') as f:
    intents = json.load(f)
with open('allwords.pickle', 'rb') as m:
    allw = pickle.load(m)
with open('tags.new', 'rb') as o:
    tags = pickle.load(o)
with open('dictres.pickle', 'rb') as z:
    res = pickle.load(z)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
with open('neuralnet.pickle', 'rb') as f:
    clf = pickle.load(f)
with open('torchdevice.pickle', 'rb') as u:
    fi = pickle.load(u)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1


    return bag
from difflib import get_close_matches
app = Flask(__name__)
@app.route('/',methods=['GET'])
def hello():
    return render_template('frontindex.html')
@app.route('/ri',methods=["POST"])
def ri():
    username=request.form["username"]
    try:
        import requests, json
        def weather(city):
            BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
            CITY = city
            API_KEY = "cfc8b78c3e56dc3a1113ead0e57db209"
            URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY
            response = requests.get(URL)
            if response.status_code == 200:
                data = response.json()
                main = data['main']
                temperature = (main['temp']) - 273.5
                temp_feel_like = main['feels_like']
                humidity = main['humidity']
                pressure = main['pressure']
                weather_report = data['weather']
                wind_report = data['wind']
                u = f"{CITY:-^35}" + " \n" + "Temperature:{:.2f}deg C".format(
                    temperature) +  " \n" + f"Humidity: {humidity}" + " \n" + f"Pressure: {pressure}" + " \n" + f"Weather Report: {weather_report[0]['description']}" + " \n" + f"Wind Speed: {wind_report['speed']}"
                return u
                # print(f"City ID: {data['id']}")
                # print("Temperature:{:.2f}deg C".format(temperature))
                # print(f"Feel Like: {temp_feel_like}")
                # print(f"Humidity: {humidity}")
                # print(f"Pressure: {pressure}")
                # print(f"Weather Report: {weather_report[0]['description']}")
                # print(f"Wind Speed: {wind_report['speed']}")
                # print(f"Time Zone: {data['timezone']}")
            else:
                # showing the error message
                u="Make a call to Kisan Call Centre"
                return u
        test = []
        sentencei = username
        sentence = sentencei.split(" ")
        X = bag_of_words(sentence, allw)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(fi)
        output = clf(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        print(tag)
        if tag == 'Weather':
            key = ['what', 'is', 'the', 'weather', 'report', 'of']
            for i in sentence:
                if i not in key:
                    u=weather(i)
        else:
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            for intent in intents['intents']:
                if tag == intent["QueryType"]:
                    test.append(intent["QueryText"])

            p = []
            p = (get_close_matches(sentencei, test))
            if len(p) == 0:
                u="Make a call to Kisan Call Centre "
            else:
                u = res[p[0]]
                print(u)
    except:
        u="error in coding"

    return render_template('backindex.html', answer=u,question=username)

if __name__ == '__main__':
    app.run(debug=True)