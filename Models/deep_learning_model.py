import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 

class KerasModel:
    """
    This is more of a wrapper class for the Keras model from playground.ipynb
    """
    def fit(self, X, y):
        cv = CountVectorizer()
        X = cv.fit_transform(X).toarray()  # make it a matrix, the network can handle (not sparse matrix)

        #this will tell how the input shape of the network needs to look like
        #print(len(cv.get_feature_names()))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_shape=[X_train.shape[1]], activation="relu"),
            tf.keras.layers.Dense(128,  activation="relu"),
            tf.keras.layers.Dense(128,  activation="relu"),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy'])

        # Model training
        self.history = self.model.fit(
            X_train, 
            y_train, 
            epochs=1, 
            batch_size=32, 
            validation_data=(X_test, y_test)
        )


    def predict_proba(self, X):
        probabilities = self.model.predict(X)
        return probabilities

    def predict(self,X):
        probabilities = self.predict_proba(X)
        labels = (probabilities > 0.5).astype(int)
        return labels

    def get_accuracy(self, X_test, y_test):
        self.loss, self.accuracy = self.model.evaluate(X_test, y_test)
        return self.accuracy

