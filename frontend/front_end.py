from flask import Flask, render_template, request
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from PIL import Image


#################################################################################
#               MAINTAINER : Ishan Khanka (ik1304) 
#################################################################################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from PIL import Image

class ConvolutionalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # 3 channel, 16 feature maps, 5x5 square convolution
        self.conv2 = nn.Conv2d(16, 128, 5) # 16 input, 128 output, 5x5 square convolution
        
        self.linear1 = nn.Linear(128 * 53 * 53, 64)  # 64 hidden units
        self.linear2 = nn.Linear(64, num_outputs)  # 64 hidden units to 10 output units

    def forward(self, input):
        output = F.tanh(self.conv1(input))
        output = F.max_pool2d(output, (2, 2))   # 2 by 2 max pooling (subsampling) 
        output = F.tanh(self.conv2(output))
        output = F.max_pool2d(output, (2, 2))   # 2 by 2 max pooling (subsampling) 
        
        # flatten to vector
        output = output.view(-1, self.num_flat_features(output)) # flatten features
        output = self.linear1(output)
        output = F.tanh(output)
        output = self.linear2(output)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def pred_predict_single_image(img1):
    # load the model and use for evaluation
    num_inputs = 150528 # 3 x 224 x 224 color images
    num_outputs = 2
    lr = 0.001 

    network = ConvolutionalNet(num_inputs, num_outputs)
    optimizer = optim.SGD(network.parameters(), lr=lr)

    network.load_state_dict(torch.load("cnn.pt")) 
    network.eval() 
    transform = transforms.ToTensor()

    # load your image(s)
    img = Image.open(app.config['UPLOAD_FOLDER']+ '/' + img1)

    # Transform
    input = transform(img)

    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input.unsqueeze(0)

    # Get prediction
    output = network(input)

    prob = F.softmax(output)[:, 1].item()
    pred = output.data.max(1, keepdim=True)[1].flatten().item() # get the index of the max log-probability

    return prob, pred

@app.route('/')
def upload_f():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        convert = ['Benign', 'Malignant']
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))

        prob, pred = pred_predict_single_image(f.filename)

        if pred == 1:
            res = str(round(prob*100, 2)) +"% "+ convert[pred]
        else:
            res = str(round((1-prob)*100, 2)) +"% "+ convert[pred]
        return res

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug = True, host= '0.0.0.0', port = port)