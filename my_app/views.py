from django.shortcuts import render

import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf

def home(request):

    return render(request, 'home.html')




#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

#Our dictionary
label_dict = {0:'Cat', 1:'Giraffe', 2:'Sheep', 3:'Bat', 4:'Octopus', 5:'Camel'}


#Initializing the Default Graph (prevent errors)
graph =tf.compat.v1.get_default_graph()

# Use pickle to load in the pre-trained model.
# with open(f'E:\\python projects\\EduKids\\my_project_copy\\modelAI\\model_cnn.pkl', 'rb') as f:
#         model = pickle.load(f)
# model=pickle.load('E:\\python projects\\EduKids\\my_project_copy\\modelAI\\model_cnn.pkl')

from tensorflow.keras.models import load_model


print(tf.__version__)
def result(request):
    global graph
    with graph.as_default():
            model = load_model("E:\\python projects\\EduKids\\my_project\\modelAI\\keras.h5")
            if request.method == 'POST':
                    final_pred = None
                    #Preprocess the image :
                    #Access the image
                    # draw = request.form['url']
                    draw = request.POST.get('url')
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    #convert into grayscale
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.set the image to 28x28 shape
                    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 28, 28,1).astype('float32')
                    #Launch prediction
                    my_prediction = model.predict(vect)
                    probabilities = model.predict_proba(vect)
                    print(probabilities)
                    #Getting the index of the maximum prediction
                    print(my_prediction)
                    index = np.argmax(my_prediction[0])
                    print(index)
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]
                    print(final_pred)
                    #image comparison
                    



                    context = {'final_pred': final_pred}
            return render(request, 'home.html', context)


