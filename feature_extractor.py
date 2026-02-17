# import necessary packages
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


class BirdFeatureExtractor:
    """Feature extractor model"""

    def __init__(self):
        self.img_height, self.img_width = 224, 224
        self.resnet_model = None
        self.classes = []
        self.files = []
        self.data = []
        self.dataset_path = "./dataset/"
        self.features = []

    def load_resnet_model(self):
        """" load the resnet model"""
        self.resnet_model = ResNet50(weights="imagenet", include_top=False, pooling='avg',
                                     input_shape=(self.img_height, self.img_width, 3))

    def get_resnet_prediction(self, input):
        """Get the resnet prediction for the image input and reshape it"""
        return self.resnet_model.predict(input).reshape(2048)

    def load_image(self, img_path):
        """Load image from the file"""
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        # Convert image pixels to numpy array
        img_array = image.img_to_array(img)
        # Expand the 3d array to 4d array
        img_array = np.expand_dims(img_array, axis=0)
        # Prepare the image for the model
        img_array = preprocess_input(img_array)
        return img_array

    def load_dataset(self):
        """Load dataset"""
        with open("./metadata/classes.txt") as f:
            # Extract the classes
            classes = f.read().splitlines()
            self.classes = {i.split('.')[0]: i.split('.')[1] for i in classes}
        with open("./metadata/files.txt") as f:
            # Extract the image path
            self.files = f.read().splitlines()
            self.data = [{"class": file.split('.')[0], "filename": self.dataset_path + file} for file in self.files]

    def extract_feature_single(self, filename):
        """Feature extraction of a single input"""
        input = self.load_image(filename)
        return self.get_resnet_prediction(input)

    def extract_features(self):
        """Looping to get the feature vectors of all the images"""
        for data_line in tqdm(self.data, total=len(self.data)):
            self.features.append(self.extract_feature_single(data_line['filename']))
        # Save the feature vectors and labels into a numpy file
        np.save('features.npy', np.vstack(self.features), allow_pickle=True, fix_imports=True)
        np.save('labels.npy', np.array([i['class'] for i in self.data]), allow_pickle=True, fix_imports=True)


if __name__ == "__main__":
    bird_feature_extractor = BirdFeatureExtractor()
    bird_feature_extractor.load_dataset()
    bird_feature_extractor.load_resnet_model()
    bird_feature_extractor.extract_features()
