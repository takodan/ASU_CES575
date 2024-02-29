import scipy.io

import numpy as np



Numpyfile = scipy.io.loadmat('mnist_data.mat')



# extract features 

train_data = Numpyfile['trX']

train_y = Numpyfile['trY'][0]

test_data = Numpyfile['tsX']

test_y = Numpyfile['tsY'][0]



#mean ans std of each row (axis = 1)

mean_data = np.mean(train_data, axis =1)

print("mean_data", mean_data)

std_data = np.std(train_data, axis =1)

print("std_data", std_data)



mean_data_test = np.mean(test_data, axis =1)

print("mean_data_test", mean_data_test)

std_data_test = np.std(test_data, axis =1)

print("std_data_test", std_data_test)



temp1 = np.reshape(mean_data, (len(mean_data), 1))

temp2 = np.reshape(std_data, (len(mean_data), 1))

train_data = np.append(temp1, temp2, axis=1)



temp1 = np.reshape(mean_data_test, (len(mean_data_test), 1))

temp2 = np.reshape(std_data_test, (len(mean_data_test), 1))

test_data = np.append(temp1, temp2, axis=1)



class NaiveBayes:

    def __init__(self):

        self.mean = {}

        self.covariance = {}

        self.inv_cov = {}

        self.determinant = {}

        self.probs = {}

    

    def train(self, train, label):

        label_8 = np.transpose(np.argwhere(label == 1))[0]

        print ("label_8", label_8)

        label_7 = np.transpose(np.argwhere(label == 0))[0]

        

        data_8 = train[label_8]

        data_7 = train[label_7]

        

        mean_8 = np.mean(data_8, axis=0)

        mean_7 = np.mean(data_7, axis=0)

        

        self.mean[1] = mean_8

        self.mean[0] = mean_7  

        print("NaiveBayes.mean", self.mean)

        

        cov_8 = np.cov(np.transpose(data_8))

        cov_8[0][1] = 0

        cov_8[1][0] = 0

        cov_7 = np.cov(np.transpose(data_7))

        cov_7[0][1] = 0

        cov_7[1][0] = 0

        self.covariance[1] = cov_8

        self.covariance[0] = cov_7

        

        self.inv_cov[1] = np.linalg.inv(cov_8)

        self.inv_cov[0] = np.linalg.inv(cov_7)

        

        self.determinant[1] = np.linalg.det(cov_8)

        self.determinant[0] = np.linalg.det(cov_7)

        

        self.probs[1] = len(data_8)/len(train)

        self.probs[0] = len(data_7)/len(train)

        

    def prob(self, inp, digit):

        temp1 = inp - self.mean[digit]

        temp2 = 0.5*np.dot(np.dot(temp1, self.inv_cov[digit]), np.transpose(temp1))

        #prob of input being digit P(digit/input)

        return self.probs[digit]*np.exp(-temp2)/((2*np.pi)*(self.determinant[digit]**0.5))

    

    def predict(self, inp):

        results = []

        for i in range(0, len(inp)):

            if self.prob(inp[i], 0) > self.prob(inp[i], 1):

                results.append(0)

            else:

                results.append(1)

        return results

    

class logistic:

    def __init__(self):

        self.weight = np.random.random(2)

        self.w0 = 0

        self.lr = 1

        self.no = 10000

    

    def sigma(self, inp):

        return 1/(1+np.exp(-inp))

    

    def calinp(self, inp):

        temp = np.dot(inp, np.transpose(self.weight))

        return temp+self.w0

    

    def loss(self, inp, y):

        temp = self.sigma(self.calinp(inp))

        loss = np.dot((y - temp), inp)/len(y)

        return loss, np.sum(y-temp)

    

    def train(self, inp, y):

         for i in range (self.no):

             loss, lossw0 = self.loss(inp, [y])

             self.weight += self.lr*loss[0]

             self.w0 += self.lr*lossw0

             if i % 500 == 0:

                 self.lr /= 2

    

    def predict(self, inp):

        results = []

        for i in range(0, len(inp)):

            if self.sigma(self.calinp(inp[i])) > 0.5:

                results.append(1)

            else:

                results.append(0)

        return results

        

        

def accuracy(predict, truth):

    count = 0

    count_7 = 0

    tot_7 = 0

    count_8 = 0

    tot_8 = 0

    for i in range(0, len(predict)):

        if predict[i] == truth[i]:

            if predict[i] == 0:

                count_7 += 1

            else:

                count_8 += 1

            count += 1

        if truth[i] == 0:

            tot_7 += 1

        else:

            tot_8 += 1

    return count/len(truth), count_7/tot_7, count_8/tot_8

    

class1 = NaiveBayes()

class1.train(train_data, train_y)

        

prediction = class1.predict(test_data)

acc = accuracy(prediction, test_y)

print("Accuracy using Naive Bayesian for overall, number 7, number 8 as follow: ", str(acc))



class2 = logistic()

class2.train(train_data, train_y)

predt = class2.predict(test_data)

acc = accuracy(predt, test_y)

print("Accuracy using Logistic Regression for overall, number 7, number 8 as follow: ", str(acc))