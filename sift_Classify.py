import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


class sift_model:

    def read_data(self, label2id):
        X = []
        Y = []
        for label in os.listdir('D:/TGMT/BTL/sifft_train/Data/train'):
            for img_file in os.listdir(os.path.join('D:/TGMT/BTL/sifft_train/Data/train', label)):
                img = cv2.imread(os.path.join('D:/TGMT/BTL/sifft_train/Data/train', label, img_file))            
                X.append(img)
                Y.append(label2id[label])
        return X, Y


    #Trich xuat dac trung
    def extract_sift_features(self, X):
        image_descriptors = []
        sift = cv2.SIFT_create()

        for i in range(len(X)):
            kp, des = sift.detectAndCompute(X[i], None)
            image_descriptors.append(des)

        return image_descriptors



    def kmeans_bow(self, all_descriptors, num_clusters):
        bow_dict = []
        kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
        bow_dict = kmeans.cluster_centers_
        return bow_dict


    #Xay dung vecto dac trung tu dict
    def create_features_bow(self, image_descriptors, BoW, num_clusters):
        X_features = []
        for i in range(len(image_descriptors)):
            features = np.array([0] * num_clusters)
            if image_descriptors[i] is not None:
                distance = cdist(image_descriptors[i], BoW)
                argmin = np.argmin(distance, axis=1)
                for j in argmin:
                    features[j] += 1
            X_features.append(features)
        return X_features

    def main(self, filename):

        # Label to id, used to convert string label to integer 
        label2id = {'left_hand':0, 'right_hand':1, 'like':2, 'win':3}
        X, Y = self.read_data(label2id)

        image_descriptors = self.extract_sift_features(X)

        #Xay dung tu dien
        all_descriptors = []
        for descriptors in image_descriptors:
            if descriptors is not None:
                for des in descriptors:
                    all_descriptors.append(des)

        num_clusters = 150#

        if not os.path.isfile('./bow_dictionary.pkl'):
            BoW = self.kmeans_bow(all_descriptors, num_clusters)
            pickle.dump(BoW, open('./bow_dictionary.pkl', 'wb'))
        else:
            BoW = pickle.load(open('./bow_dictionary.pkl', 'rb'))

        X_features = self.create_features_bow(image_descriptors, BoW, num_clusters)

        #Xay dung model
        X_train = [] 
        X_test = []
        Y_train = []
        Y_test = []
        X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.5)

        svm = sklearn.svm.SVC(kernel='linear', C=30, gamma='auto')#
        svm.fit(X_train, Y_train)

        #Thu predict 
        img_init = cv2.imread(filename)
        img_init = cv2.resize(img_init, (640, 840))
        img_test = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
        _ , img_test = cv2.threshold(img_test, 135, 255, cv2.THRESH_BINARY)#
        img = [img_test]
        img_sift_feature = self.extract_sift_features(img)
        img_bow_feature = self.create_features_bow(img_sift_feature, BoW, num_clusters)
        img_predict = svm.predict(img_bow_feature)

        # print(img_predict)
        
        for key, value in label2id.items():
            if value == img_predict[0]:
                res = key
                # print('Your prediction: ', key , ' ' , svm.score(X_test, Y_test))

        return res

            #Accuracy
            #print(svm.score(X_test, Y_test))

        #Show image
        # cv2.imshow("Img", img_test)
        # cv2.waitKey()


