# import necessary packages
import numpy as np
from keras.layers import Dense
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    """Neural Network classifier"""

    def __init__(self):
        """Load the data"""
        self.features = np.load('features.npy')
        self.labels = np.load('labels.npy')
        # Create a label encoder object
        self.label_encoder = preprocessing.LabelEncoder()
        # Perform PCA on the features
        self.pca = PCA(n_components=1024)
        self.y = self.label_encoder.fit_transform(self.labels)
        self.X = self.pca.fit(self.features).transform(self.features)
        # Split the data into testing and training sets
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.2,
                                                                            random_state=0)

    def load_model(self):
        """Load the neural network model"""
        self.model = Sequential()
        # Add input,middle and output layers
        self.model.add(Dense(1024, input_shape=(1024,), activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(200, activation='softmax', name='output'))
        # Compile the model
        self.model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.Xtrain, self.ytrain, verbose=1, batch_size=500, epochs=25)
        # Evaluate the model
        result = self.model.evaluate(self.Xtest, self.ytest)
        print(result)


if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.load_model()
