# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from methods import *
from scipy import ndimage as im


class KPCA(object):


    def __init__(self):
        pass

    @staticmethod
    def train(type, data):
        # Manage Images
        images_directory = data['images_dir']; area = data['v_size'] * data['h_size']
        images_per_person = data['images_quantity_per_person']; number_of_people = data['people_quantity']
        training_n = data['training_n']; test_n = data['test_n']
        subjects = [f for f in listdir(images_directory) if isdir(join(images_directory, f))]
        images = np.zeros([training_n * number_of_people, area])
        imagetst = np.zeros([test_n * number_of_people, area])
        training_image = 0; test_image = 0; person_image = 0; subject_number = 0; training_names = []; test_names = []
        for subject in subjects:
            for k in range(1, images_per_person + 1):
                a = im.imread(images_directory + '/' + subject + '/{}'.format(k) + '.pgm')
                if person_image < training_n:
                    images[training_image, :] = (np.reshape(a, [1, area]) - 127.5) / 127.5
                    training_names.append(str(subject))
                    training_image += 1
                else:
                    imagetst[test_image, :] = (np.reshape(a, [1, area]) - 127.5) / 127.5
                    test_names.append(str(subject))
                    test_image += 1
                person_image += 1
            subject_number += 1
            if subject_number > number_of_people - 1:
                break
            person_image = 0

        degree = 2
        total_pic = data['training_n'] * data['people_quantity']
        total_test_pic = data['test_n'] * data['people_quantity']
        K = (np.dot(images, images.T) / total_pic + 1) ** degree
        unoM = np.ones([total_pic, total_pic]) / total_pic
        K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))

        ###################
        # Get Eigenvalues #
        ###################

        w, alpha = getEigenValues(K)
        lambdas = w
        lambdas = np.flipud(lambdas)
        alpha = np.fliplr(alpha)

        for col in range(alpha.shape[1]):
            alpha[:, col] = alpha[:, col] / np.sqrt(abs(lambdas[col]))

        improypre = np.dot(K.T, alpha)
        uno_ml = np.ones([total_test_pic, total_pic]) / total_pic
        k_test = (np.dot(imagetst, images.T) / total_pic + 1) ** degree
        k_test = k_test - np.dot(uno_ml, K) - np.dot(k_test, unoM) + np.dot(uno_ml, np.dot(K, unoM))
        im_test_projection_pre = np.dot(k_test, alpha)

        nmax = alpha.shape[1]
        nmax = 100
        accs = np.zeros([nmax, 1])

        if type == 'test':
            print "Testing"
            for neigen in range(1, nmax):
                improy = improypre[:, 0:neigen]
                imtstproy = im_test_projection_pre[:, 0:neigen]
                clf = svm.LinearSVC()
                clf.fit(improy, training_names)
                accs[neigen] = clf.score(imtstproy, test_names)
                print('Precisión con {0} autocaras: {1} %\n'.format(neigen, accs[neigen] * 100))

            fig, axes = plt.subplots(1, 1)
            axes.semilogy(range(nmax), (1 - accs) * 100)
            axes.set_xlabel('No. autocaras')
            axes.grid(which='Both')
            fig.suptitle('Error')

        elif type == 'predict':
            print "Predicting"
            picture = im.imread(data['path'])
            fig, axes = plt.subplots(1, 1)
            axes.imshow(picture, cmap='gray')
            fig.suptitle('Image to predict')
            plt.show()
            picture = np.reshape((picture - 127.5) / 127.5, [1, data['h_size'] * data['v_size']])

            improy = improypre[:, 0:60]
            imtstproy = im_test_projection_pre[:, 0:60]

            clf = svm.LinearSVC()
            clf.fit(improy, training_names)

            improypre = np.dot(K.T, alpha)
            uno_ml = np.ones([1, total_pic]) / total_pic
            k_test = (np.dot(picture, images.T) / total_pic + 1) ** degree
            k_test = k_test - np.dot(uno_ml, K) - np.dot(k_test, unoM) + np.dot(uno_ml, np.dot(K, unoM))
            im_test_projection_pre = np.dot(k_test, alpha)
            picture_projection = im_test_projection_pre[:,0:60]

            sub = clf.predict(picture_projection)[0]
            print("Subject is: {} \n".format(sub))

            picture = im.imread(images_directory + '/' + sub + '/1.pgm')
            fig, axes = plt.subplots(1, 1)
            axes.imshow(picture, cmap='gray')
            fig.suptitle('Subject Predicted')
            plt.show()
    @staticmethod
    def test(data):
        KPCA.train('test', data)

    @staticmethod
    def predict(data):
        KPCA.train('predict', data)


