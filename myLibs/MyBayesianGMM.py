# this function is based the Naive Bayesian Gaussian Model for two classes classification problem
# the likelihood is calculated by the log-likelihood which is for high dimension features and normalized vector

import numpy as np
import math

class BayesianGMMClassifier(object):

    @classmethod

    def __init__(self, parameters=None):
        self.parameters = parameters

    def normpdf(self, x, mu, sigma):
        if sigma<10e-10:
            sigma=10e-5
        var = float(sigma)**2
        pi = 3.1415926
        num = (math.exp(-(float(x)-float(mu))**2/(2*var)))/(2*pi*var)**.5
        return num

    def fit(self, data, label):
        """
        Train the classifier by using training data
        """

        obj=data[label>0,:]
        bkg=data[label==0,:]

        obj_mu=obj.mean(0)
        obj_sigma=obj.std(0)

        bkg_mu=bkg.mean(0)
        bkg_sigma=bkg.std(0)

        self.model = {'obj_mu': obj_mu, 'obj_sigma': obj_sigma, 'bkg_mu': bkg_mu, 'bkg_sigma': bkg_sigma}

        return self

    def predict_proba(self, test_data):
        """
        Predict the class labels in probability for the provided test data
        :param test_data(n data samples, m features)
        :return: probability estimation for two classes
        """

        try:
            getattr(self, "model")
        except AttributeError:
            raise RuntimeError("Please train your classifier/model")

        obj_mu=self.model.get('obj_mu')
        obj_sigma=self.model.get('obj_sigma')

        bkg_mu=self.model.get('bkg_mu')
        bkg_sigma=self.model.get('bkg_sigma')

        obj_prob=np.zeros(len(test_data[:,0]))
        bkg_prob=np.zeros(len(test_data[:,0]))

        for n in range(len(test_data[:,0])): # n data samples
            for m in range(len(test_data[0,:])): # m features
                # print (normpdf(test_data[n,m], obj_mu[m], obj_sigma[m]))
                obj_prob[n]=obj_prob[n]+math.log(self.normpdf(test_data[n,m], obj_mu[m], obj_sigma[m])+10e-10)
                bkg_prob[n]=bkg_prob[n]+math.log(self.normpdf(test_data[n,m], bkg_mu[m], bkg_sigma[m])+10e-10)

        # calculate the log-likelihood ratio and normalize to [0 1] probabilities
        prob=obj_prob-bkg_prob
        prob=(prob-prob.min())/(prob.max()-prob.min())

        # for two classes
        prob_k=[]
        for i in range(len(prob)):
            prob_k.append([1-prob[i], prob[i]])
        return np.array(prob_k)

    def predict_wproba(self, model, test_data, weights):
        """
        Predict the class labels in probability for the provided test data
        :param test_data(n data samples, m features), and weight for each feature
        :return: probability estimation for two classes
        """
        obj_mu=model.get('obj_mu')
        obj_sigma=model.get('obj_sigma')

        bkg_mu=model.get('bkg_mu')
        bkg_sigma=model.get('bkg_sigma')

        obj_prob=np.zeros(len(test_data[:,0]))
        bkg_prob=np.zeros(len(test_data[:,0]))

        # print test_data.shape

        for n in range(len(test_data[:,0])): # number of samples
            for m in range(len(test_data[0,:])):  # number of features
                # print (normpdf(test_data[n,m], obj_mu[m], obj_sigma[m]))
                obj_prob[n]=obj_prob[n]+weights[m]*math.log(self.normpdf(test_data[n,m], obj_mu[m], obj_sigma[m])+10e-10)
                bkg_prob[n]=bkg_prob[n]+weights[m]*math.log(self.normpdf(test_data[n,m], bkg_mu[m], bkg_sigma[m])+10e-10)

        prob=obj_prob-bkg_prob
        prob=(prob-prob.min())/(prob.max()-prob.min())

        # for two classes
        prob_k=[]
        for i in range(len(prob)):
            prob_k.append([1-prob[i], prob[i]])
        return np.array(prob_k)