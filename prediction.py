# import necessary packages
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from feature_extractor import BirdFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold


class BirdClassifier:
    def __init__(self):
        """Classifier model"""
        """Load the data"""
        self.features = np.load('features.npy')
        self.labels = np.load('labels.npy')
        # Perform PCA on the features
        self.pca = PCA(n_components=1024)
        self.pca.fit(self.features)
        self.X = self.pca.transform(self.features)
        # Train-test split
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.labels, random_state=0)
        self.classifier = None

    def load_models(self):
        """Create classifier object"""
        self.lr_classifier = LogisticRegression()
        self.knn_classifier = KNeighborsClassifier(n_neighbors=10)
        self.dtree_classifier = DecisionTreeClassifier()
        self.svc_classifier = SVC()
        # Train the model using the training set
        self.lr_classifier.fit(self.Xtrain, self.ytrain)
        self.knn_classifier.fit(self.Xtrain, self.ytrain)
        self.dtree_classifier.fit(self.Xtrain, self.ytrain)
        self.svc_classifier.fit(self.Xtrain, self.ytrain)

    def get_boxplot(self):
        models = []
        models.append(("Logistic Regression", self.lr_classifier))
        models.append(("KNN", self.knn_classifier))
        models.append(("Decision Tree", self.dtree_classifier))
        models.append(("SVC", self.svc_classifier))
        results = []
        names = []
        for name, model in models:
            kfold = KFold(n_splits=10)
            cv_results = cross_val_score(model, self.Xtest, self.ytest, cv=kfold, scoring="accuracy")
            results.append(cv_results)
            names.append(name)

        # Boxplot with Score of each model
        fig = plt.figure()
        fig.suptitle('Performance of each model')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    def load_label_names(self):
        """Load the label names"""
        with open("./metadata/classes.txt") as f:
            class_names = f.read().splitlines()
            self.class_names = {i.split('.')[0]: i.split('.')[1] for i in class_names}


if __name__ == "__main__":
    bird_classifier = BirdClassifier()
    bird_feature_extractor = BirdFeatureExtractor()
    bird_feature_extractor.load_resnet_model()
    bird_feature_extractor.load_dataset()
    bird_classifier.load_label_names()
    bird_classifier.load_models()

    # Read the image to be predicted
    img_disp = cv.imread("Bobolink.jpg")
    # Resize the image
    img_disp = cv.resize(img_disp, (224, 224))
    # Prepare the image for the resnet model using the load_image function from feature extractor model
    img = bird_feature_extractor.load_image("Bobolink.jpg")
    # Get the resnet prediction for the image
    predicted_features = bird_feature_extractor.get_resnet_prediction(img)
    # Perform PCA on the feature vector
    predicted_features = bird_classifier.pca.transform(predicted_features.reshape(1, -1))
    # Predict the class of the image using predict function in the classifer model
    predicted_label = bird_classifier.lr_classifier.predict(predicted_features.reshape(1, -1))
    font = cv.FONT_HERSHEY_SIMPLEX
    # Decode the label to get the class name
    predicted_label_decoded = bird_classifier.class_names[predicted_label[0]]
    # Display the image with the class label on it
    img_disp = cv.putText(img_disp, 'Predicted:' + predicted_label_decoded, (10, 210), font, 0.5, (0, 0, 255), 1,
                          cv.LINE_AA)
    cv.imshow('Prediction', img_disp)
    cv.waitKey(0)
