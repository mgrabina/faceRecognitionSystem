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

from library.utils import getImages
from methods import *
from scipy import ndimage as im

class PCA(object):

    trained = False

    @staticmethod
    def train(type, data):
        if PCA.trained == False:
            PCA.trained = True
            mypath = '../att_faces/'

            # image size
            horsize = 92
            versize = 112
            areasize = horsize * versize

            # number of figures
            personno = 40
            trnperper = 6
            tstperper = 4
            trnno = personno * trnperper
            tstno = personno * tstperper

            # TRAINING SET
            images = np.zeros([trnno, areasize])
            person = np.zeros([trnno, 1])
            imno = 0
            per = 0
            onlydirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]
            images, imagetst, trainingNames, testNames = getImages("../att_faces/", 112 * 92,10, 40,6, 4)

            # CARA MEDIA
            meanimage = np.mean(images, 0)
            fig, axes = plt.subplots(1, 1)
            axes.imshow(np.reshape(meanimage, [versize, horsize]) * 255, cmap='gray')
            fig.suptitle('Imagen media')

            # resto la media
            images = [images[k, :] - meanimage for k in range(images.shape[0])]
            imagetst = [imagetst[k, :] - meanimage for k in range(imagetst.shape[0])]

            # PCA
            images_matrix = np.asmatrix(images)
            S, V = getSingluarValuesAndEigenVectors(images_matrix)

            nmax = V.shape[0]
            nmax = 100
            accs = np.zeros([nmax, 1])

        if type == 'test':
            print "Testing..."
            for neigen in range(1, nmax):
                # Me quedo sólo con las primeras autocaras
                B = V[0:neigen, :]
                # proyecto
                improy = np.dot(images, np.transpose(B))
                imtstproy = np.dot(imagetst, np.transpose(B))

                # SVM
                # entreno
                clf = svm.LinearSVC()
                clf.fit(improy, person.ravel())
                accs[neigen] = clf.score(imtstproy, persontst.ravel())
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

            B = V[0:60, :]
            improy = np.dot(images, np.transpose(B))
            clf = svm.LinearSVC()
            clf.fit(improy, trainingNames)
            picture -= meanimage
            pictureProy = np.dot(picture, B.T)
            print("Subject is: {} \n".format(clf.predict(pictureProy)[0]))
        else:
            print "Error"

    @staticmethod
    def test():
        PCA.train('test', {})

    @staticmethod
    def predict(path, data):
        PCA.train('predict', data)

