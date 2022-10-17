
https://user-images.githubusercontent.com/103250661/190868849-1fad59e4-c2f0-4baf-a26a-1003eda841bc.mp4


# Fashion Recommended System

- A website where a user can upload an image (of a dress, watch, or shoe) and get products that are similar (visually) to the supplied image.

Working
-

![working](https://user-images.githubusercontent.com/103250661/196235150-1c0b0e75-f674-4188-a4c5-8cc4d30813a9.png)
Predicting
-


![predicting](https://user-images.githubusercontent.com/103250661/196235189-b6382535-c613-406c-b03a-b0b4bf0a1267.png)

- Images are taken from the kaggle  [dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small).


Step -1 : Import Model
-
- First I imported a  CNN model, named ResNET50 having high performance and accuracy, built in Keras Module.

```
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

```
-> Taking the weights trained on "imagenet" dataset, creating own top layer, size of image(224,224,3) which is the standard size of image.


-> I did not train the model, so ``` model.trainable = False ``` , since I am using weights trained on "imagenet" dataset.

-> Now added top layer using GlobalMaxPooling2D(), to change the default shape of model  ``` from (None, 7, 7, 208 ) to (None, 2048) ```


step -2 Extract Features
-
- Comparing one image with a large dataset of images pixel by pixel is difficult, which is ```224 x 224 x 3 x No. of images ```in dataset.
- So, I extracted features(2048) of every image from the dataset. Now, the comparision is now according to the feature by feature, reduced to ```2048 x No. of images```.


``` 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from numpy.linalg import norm



def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


```

-> First loaded the image and converted into a numpy array whose shape is 224,224,3.

-> Used np.expand_dims to convert array into 4D since keras library works on batch of images.

-> By using preprocess_input() function, the images are converted from RGB to BGR, then each color channel is zero-centerd with respect to imageNet dataset, without scaling.

-> With predict() function the image is given to the ResNet50 model and flatten() to get 2048 features of the image.

-> Normalized all the values of the features to range from 0 to 1, using norm() function.

Step 3: Export features
-
- After extracting features of all images, I stored those into a file to use in the next step using pickle.dump(). Now this file when coverted into a numpy array, it is the requrired  2D array with ``` No. of images x 2048 features```.

- Also  stored the names of all the images in a file.
``` 
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))


```

Final step 4: Generating Recomendations
-
- In this step, I took a new_image and extracted features using the above model. Then, I plotted the feature_list created in the above step and the new_image features in 2048 dimensions and returned the 5 closest vectors i.e., those image vectors which are closest to new_image vector.

```
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([features])

```

-> To calculate the distance between new_image and the feature_list, I used NearestNeighbors algorithm.

-> Instantiated with the no. of neighbors = 5, used brute force algorithm and calculated euclidean distance.

-> It returns the distance and indices. These indices are used to get the images from the dataset.
