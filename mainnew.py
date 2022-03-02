import numpy as np
import nltk
import random
nltk.download('punkt')
import pickle
import json

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
with open('C:\\Users\\anand\\PycharmProjects\\NAAC\\2-1.json', 'r') as f:
    intents = json.load(f)
ignore_words=['?','!','.',',','(',')','&','@']
all_words = []
tags = []
xy = []
patternize = []
processed_patternize = []
answer = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['QueryType']  # tag=intent eg-Fertilizer,market price,cultivation,etc..
    ans = intent['KccAns']
    bname = intent['BlockName']  # answers for the query text
    answer.append(ans)
    # add to tag list
    tags.append(tag)
    pattern = intent['QueryText']  # querytext
    patternize.append(pattern)
    # tokenize each word in the sentence
    w = pattern.split(" ")
    w.append(bname)

    # add to our words list
    all_words.extend(w)
    i = w
    i = [stem(k) for k in i if
         k not in ignore_words]  # i) removing punctuation words from ignore_words  ii) finding root words using stemming operation
    i = " ".join(i)
    processed_patternize.append(i)
    # add to xy pair
    xy.append((w, tag))
print("tokenized---------------------------")
print(xy)
y_train_1 = tags
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))
with open('tags.new', 'wb') as o:
    pickle.dump(tags, o)
remove_words=['(',')','&']          #  removing the symbols in (,),&
all_wordsn=[]
for i in all_words:
  if i[0] in remove_words or i[-1] in remove_words:
    if i[0] in remove_words:
      i=i[1:]
    if i[-1] in remove_words:
      i=i[:-1]
      all_wordsn.append(i)
  else:
    all_wordsn.append(i)
all_words=all_wordsn
print("alwords------------------")
print(all_words)
X_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)     #all_words is a dictionary now.
    X_train.append(bag)
print("xtrain------------------")
print(X_train)
with open('allwords.pickle', 'wb') as m:
    pickle.dump(all_words, m)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train_1)
list(le.classes_)
y_train = le.transform(y_train_1)
X_train = np.array(X_train)
y_train = np.array(y_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33,random_state=42)
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
dataset = ChatDataset()
batch_size=8
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)
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
# Hyper-parameters
num_epochs = 20
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
with open('torchdevice.pickle', 'wb') as n:
    pickle.dump(device, n)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics = "accuracy"

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
FILE = "models/data.pth"
torch.save(data, FILE)
import pandas
print(f'training complete. file saved to {FILE}')
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(FILE)
res={}
for cl in range(0,len(patternize)):
  res.update({patternize[cl]:answer[cl]})
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
import pickle
with open('neuralnet.pickle', 'wb') as v:
    pickle.dump(model, v)
from difflib import get_close_matches
res={}
for cl in range(0,len(patternize)):
  res.update({patternize[cl]:answer[cl]})
with open('dictres.pickle', 'wb') as p:
    pickle.dump(res, p)
bot_name = "Sinegalatha"
print("Let's chat! (type 'quit' to exit)")
test = []
try:
    while True:
        sentencei = input("You: ")
        if sentencei == "quit":
            break
        sentence = sentencei.split(" ")
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        for intent in intents['intents']:
            if tag == intent["QueryType"]:
                test.append(intent["QueryText"])

        p = []
        p = (get_close_matches(sentencei, test))
        if len(p) == 0:
            print("Make a call to Kisan Call Centre ")
        else:
            u = res[p[0]]
            print(u)
except:
    print("Make a call to Kisan Call Centre ")
predict_tag=[]
for X in X_test:
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    print(predicted.item())
    predict_tag.append(predicted.item())
predict_train = np.array(predict_tag)
test_train = np.array(y_test)
from sklearn.metrics import accuracy_score
a=accuracy_score(predict_train, test_train)
print(a)