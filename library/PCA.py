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


class PCA(object):


    def __init__(self):
        pass

    def train(self):
        mypath = '../att_faces/'
        onlydirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

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
        self.person = np.zeros([trnno, 1])
        imno = 0
        per = 0
        for dire in onlydirs:
            for k in range(1, trnperper + 1):
                a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm') / 255.0
                images[imno, :] = np.reshape(a, [1, areasize])
                self.person[imno, 0] = per
                imno += 1
            per += 1

        # TEST SET
        imagetst = np.zeros([tstno, areasize])
        self.persontst = np.zeros([tstno, 1])
        imno = 0
        per = 0
        for dire in onlydirs:
            for k in range(trnperper, 10):
                a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm') / 255.0
                imagetst[imno, :] = np.reshape(a, [1, areasize])
                self.persontst[imno, 0] = per
                imno += 1
            per += 1

        # CARA MEDIA
        meanimage = np.mean(images, 0)
        fig, axes = plt.subplots(1, 1)
        axes.imshow(np.reshape(meanimage, [versize, horsize]) * 255, cmap='gray')
        fig.suptitle('Imagen media')

        # resto la media
        self.images = [images[k, :] - meanimage for k in range(images.shape[0])]
        self.imagetst = [imagetst[k, :] - meanimage for k in range(imagetst.shape[0])]

        # PCA
        images_matrix = np.asmatrix(images)
        self.S, self.V = getSingluarValuesAndEigenVectors(images_matrix)

        self.nmax = self.V.shape[0]
        self.nmax = 100
        self.accs = np.zeros([self.nmax, 1])

    def test(self):
        print "Starting"
        self.train()
        for neigen in range(1, self.nmax):
            # Me quedo sólo con las primeras autocaras
            B = self.V[0:neigen, :]
            # proyecto
            improy = np.dot(self.images, np.transpose(B))
            imtstproy = np.dot(self.imagetst, np.transpose(B))

            # SVM
            # entreno
            clf = svm.LinearSVC()
            clf.fit(improy, self.person.ravel())
            self.accs[neigen] = clf.score(imtstproy, self.persontst.ravel())
            print('Precisión con {0} autocaras: {1} %\n'.format(neigen, self.accs[neigen] * 100))

        fig, axes = plt.subplots(1, 1)
        axes.semilogy(range(self.nmax), (1 - self.accs) * 100)
        axes.set_xlabel('No. autocaras')
        axes.grid(which='Both')
        fig.suptitle('Error')

    def predict(self, image):
        self.train()


