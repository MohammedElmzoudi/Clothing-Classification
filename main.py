import numpy as np
from tqdm import tqdm
import copy
import mnist_reader
from scipy.special import expit
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from mlxtend.data import loadlocal_mnist

np.random.seed(0)

class NeuralNetwork:
    def __init__(self):


        #self.input_values, self.output_values = mnist_reader.load_mnist('Data/', kind='train')
        self.input_values, self.o = loadlocal_mnist(images_path='Data/Digit_MNIST/train-images-idx3-ubyte', labels_path='Data/Digit_MNIST/train-labels-idx1-ubyte')
        self.input_values = self.input_values.T

        #self.input_values_test, self.output_values_test = mnist_reader.load_mnist('Data/', kind='test')
        self.input_values_test, self.ot = loadlocal_mnist(images_path='Data/Digit_MNIST/t10k-images-idx3-ubyte', labels_path='Data/Digit_MNIST/t10k-labels-idx1-ubyte')
        self.input_values_test = self.input_values_test.T

        self.og_inputs = copy.copy(self.input_values_test)
        self.output_values = np.zeros([10,len(self.o)])
        self.output_values_test = np.zeros([10,len(self.ot)])

        for i in range(len(self.o)):
            self.output_values[:, i] = mnist_reader.to_vector(self.o[i])
        for i in range(len(self.ot)):
            self.output_values_test[:, i] = mnist_reader.to_vector(self.ot[i])

        self.mini_batch_inputs = 0
        self.mini_batch_outputs = 0

        n_input = 784
        n_hidden1 = 113
        n_output = 10
        n_hidden2 = 80

        self.m = self.input_values.shape[1]
        self.m_test = self.input_values_test.shape[1]
        self.e = 0.00001
        self.a = 0.01
        self.reg = 0.2

        self.weights1 = np.random.normal(scale = 0.1, size = (n_hidden1,n_input))
        #self.weights2 = np.random.normal(scale = 0.1, size = (n_hidden2,n_hidden1))
        self.weights2 = np.random.normal(scale = 0.1, size = (n_output,n_hidden1))

        self.b1 = np.random.normal(scale = 0.1, size = (n_hidden1,1))
        #self.b2 = np.random.normal(scale = 0.1, size = (n_hidden2,1))
        self.b2 = np.random.normal(scale = 0.1, size = (n_output,1))



    def iterate(self, gradient_check,e_type):

        #Randomly select 32 training samples to be used in the current training iteration
        rand_int = np.random.randint(low=0,high=self.input_values.shape[1],size = 512)
        self.mini_batch_inputs = self.input_values[:,rand_int]
        self.mini_batch_outputs = self.output_values[:,rand_int]
        self.m = self.mini_batch_inputs.shape[1]

        z_hidden_layer1 = np.matmul(self.weights1, self.mini_batch_inputs) + self.b1
        hidden_layer1 = self.relu(z_hidden_layer1)
        z_output_layer = np.matmul(self.weights2, hidden_layer1) + self.b2
        output_layer = expit(z_output_layer)

        output_layer_delta = (output_layer - self.mini_batch_outputs)

        if e_type == 'SE':
            output_layer_delta = (output_layer - self.mini_batch_outputs) * self.sigmoid_derivative(z_output_layer)
        if e_type == 'Logistic':
            output_layer_delta = (output_layer - self.mini_batch_outputs)

        hidden_layer1_delta = np.matmul(self.weights2.T, output_layer_delta)
        hidden_layer1_delta = hidden_layer1_delta * self.relu_derivative(z_hidden_layer1)

        weights2_gradient = np.matmul(output_layer_delta, hidden_layer1.T)
        weights1_gradient = np.matmul(hidden_layer1_delta, self.mini_batch_inputs.T)
        weights2_gradient = (1 / self.m) * (weights2_gradient + self.reg * self.weights2)
        weights1_gradient = (1 / self.m) * (weights1_gradient + self.reg * self.weights1)

        b1_gradient = (1/self.m) * np.sum(hidden_layer1_delta, axis=1)
        b1_gradient = np.expand_dims(b1_gradient,axis= 1)

        b2_gradient = (1/self.m) * np.sum(output_layer_delta, axis=1)
        b2_gradient = np.expand_dims(b2_gradient,axis=1)

        if gradient_check:

            w = self.unroll_parameters(self.weights1, self.weights2,self.b1,self.b2)


            ws = self.weights1.shape[0] * self.weights1.shape[1]
            we = ws+self.weights2.shape[0] * self.weights2.shape[1]
            weights2_approx = np.zeros(3)
            for i in tqdm(range(3)):
                weights2_ePlus = copy.copy(w)
                weights2_ePlus[ws+i] += self.e
                weights2_eMinus = copy.copy(w)
                weights2_eMinus[ws+i] -= self.e
                weights2_approx[i] = (self.error_gc(weights2_ePlus,e_type) - self.error_gc(weights2_eMinus,e_type)) / (2 * self.e)

            bs = we
            be = bs + len(self.b1)
            b1_approx = np.zeros(3)

            for i in tqdm(range(3)):
                b1_ePlus = copy.copy(w)
                b1_ePlus[bs+i] += self.e
                b1_eMinus = copy.copy(w)
                b1_eMinus[bs+i] -= self.e
                b1_approx[i] = (self.error_gc(b1_ePlus,e_type) - self.error_gc(b1_eMinus,e_type)) / (2 * self.e)

            print("Numerically calculated weight gradients")
            print(np.ravel(weights2_approx))
            print("Back propagation calculated weight gradients")
            print(np.ravel(weights2_gradient)[:3])

            print("Numerically calculated bias gradients")
            print(np.ravel(b1_approx))
            print("Back propagation calculated gradients")
            print(np.ravel(b1_gradient)[:3])


        self.weights1 = self.weights1 - self.a * weights1_gradient
        self.weights2 = self.weights2 - self.a * weights2_gradient


        self.b1 = self.b1 - self.a * b1_gradient
        self.b2 = self.b2 - self.a * b2_gradient

        return self.unroll_parameters(self.weights1,self.weights2,self.b1,self.b2)

    def relu_derivative(self, z):

        t = copy.copy(z)
        t[t<0] = 0
        return t

    def sigmoid_derivative(self,z):
        return expit(z) * (1 - expit(z))

    def relu(self, z):
        return np.maximum(np.zeros(z.shape),z)

    def predict(self, inputs, parameters):
        weights_1,weights_2,b1,b2 = self.roll_parameters(parameters)

        z_hidden_layer1 = np.matmul(weights_1, inputs) + b1
        hidden_layer1 = self.relu(z_hidden_layer1)

        z_output_layer = np.matmul(weights_2, hidden_layer1) + b2
        output_layer = expit(z_output_layer)

        return output_layer

    def error(self, parameters, t):
        h = self.predict(self.input_values_test, parameters)
        y = copy.copy (self.output_values_test)
        weights1_length = self.weights1.shape[0] * self.weights1.shape[1]
        weights2_length = self.weights2.shape[0] * self.weights2.shape[1]
        error = 0
        if t == 'SE':

            error = (1 / self.m_test) * np.sum((np.sum( (1 / 2) * ((y - h) ** 2) , axis=0)))
            error += (self.reg / (2 * self.m_test)) * sum(parameters[:weights1_length + weights2_length]**2)
        if t == 'Logistic':
            error = (-y* np.log(h)) - ((1-y)*np.log(1-h))
            error = (1/self.m_test) * np.sum( np.sum(  error  , axis=0))
            error += (self.reg / (2* self.m_test)) * sum(parameters[:weights1_length+weights2_length]**2)

        return error

    def error_gc(self, parameters, t):
        h = self.predict(self.mini_batch_inputs, parameters)
        y = copy.copy(self.mini_batch_outputs)
        weights1_length = self.weights1.shape[0] * self.weights1.shape[1]
        weights2_length = self.weights2.shape[0] * self.weights2.shape[1]
        error = 0
        if t == 'SE':
            error = (1 / self.m) * np.sum((np.sum((1 / 2) * ((y - h) ** 2), axis=0)))
            error += (self.reg / (2 * self.m)) * sum(parameters[:weights1_length + weights2_length] ** 2)

        if t == 'Logistic':
            error = (-y * np.log(h)) - ((1 - y) * np.log(1 - h))
            error = (1 / self.m) * np.sum(np.sum(error, axis=0))
            error += (self.reg / (2 * self.m)) * sum(parameters[:weights1_length + weights2_length] ** 2)

        return error

    def set_reg(self, r):
        self.reg = r

    def train(self, num_iter,m , gradient_check, save_loc,e_type, save_error):

        error_hist = np.zeros([num_iter])
        updated_hyperparameters = 0

        self.input_values = self.input_values[:,:m]
        self.output_values = self.output_values[:,:m]
        self.m = m


        for i in tqdm(range(num_iter)):
            if gradient_check:
                if i == 20:
                    self.iterate(True,e_type)

            updated_hyperparameters = self.iterate(False,e_type)
            if save_error:
                error_hist[i] = ann.error(updated_hyperparameters,'Logistic')

        if save_error:
            plt.plot(error_hist)
            plt.ylabel("Cost")
            plt.xlabel("Iterations")
            plt.show()

        if save_loc != "":
            np.save(save_loc, updated_hyperparameters)

        return updated_hyperparameters



    def mean_std(self,X):
        return (X - np.mean(X)) / np.std(X)

    def preprocess_data(self):
        self.input_values_test = self.mean_std(self.input_values_test)
        self.input_values = self.mean_std(self.input_values)

    def predict_test(self, index,  parameters):
        test_input = np.expand_dims(self.input_values_test[:,index],axis=1)
        predicted_val = self.predict(test_input,parameters)
        print("Predicted Value:")
        print(np.round(np.ravel(predicted_val),1 ))
        print("Actual Value:")
        print(self.output_values_test[:,index])

        im_data = np.reshape(self.og_inputs[:,index],[28,28])
        im = Image.fromarray(im_data,'L')
        im.show()
        return predicted_val

    def predict_image(self, image_loc, parameters):

        test_input,f_image = self.convert_image(image_loc)

        predicted_val = self.predict(test_input,parameters)
        print("Predicted Value:")
        print(np.round(np.ravel(predicted_val),5 ))
        f_image.show()

        return predicted_val

    def convert_image(self,image_loc):

        f_image = Image.open(image_loc).convert('L')
        f_image = f_image.resize((28,28), Image.ANTIALIAS)
        f_image = PIL.ImageOps.invert(f_image)
        image = np.asarray(f_image)

        image = self.mean_std(image)
        Image.fromarray(image,'L').show()
        image = np.reshape(image, (image.shape[1] * image.shape[0], 1))

        return image, f_image

    def unroll_parameters(self,weights1,weights2,bias_1,bias_2):

        weights1 = np.ravel(weights1)
        weights2 = np.ravel(weights2)
        bias_1 = np.ravel(bias_1)
        bias_2 = np.ravel(bias_2)

        rolled_parameters = np.append(np.append(weights1,weights2),np.append(bias_1,bias_2))
        return rolled_parameters

    #[inclusive:exclusive]
    def roll_parameters(self, parameters):
        weights1_length = self.weights1.shape[0]*self.weights1.shape[1]
        weights2_length = self.weights2.shape[0]*self.weights2.shape[1]
        bias1_length = len(self.b1)
        bias2_length = len(self.b2)
        weights1 = parameters[: weights1_length]
        weights2 = parameters[weights1_length:weights1_length+weights2_length]
        bias1 = parameters[weights1_length+weights2_length:weights1_length+weights2_length+bias1_length ]
        bias2 = parameters[weights1_length+weights2_length+bias1_length:weights1_length+weights2_length+bias1_length+bias2_length]

        weights1 = np.reshape(weights1,self.weights1.shape)
        weights2 = np.reshape(weights2, self.weights2.shape)
        bias1 = np.reshape(bias1,self.b1.shape)
        bias2 = np.reshape(bias2,self.b2.shape)

        return weights1,weights2,bias1,bias2

if __name__ == "__main__":

    ann = NeuralNetwork()
    #p = np.load("trained_weights/wb_60k_10I.npy")
    #p = np.load("Data/parameters_60_10_reg_ms.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_128.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_256.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_512.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_1024.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_512_tanh.npy")
    #p = np.load("Data/parameters_60_10_reg_ms_schotastic_512_relu.npy")
    p = np.load("Data/mnist_parameters_60_10_reg_ms_schotastic_512_relu.npy")


    ann.preprocess_data()
    ann.set_reg(0.25)
    #Iterations, M, GradientCheck, Saving location, Error function, Record error
    #ann.train(10000,60000,False, "Data/mnist_parameters_60_10_reg_ms_schotastic_512_relu", 'Logistic', False)
    #print(ann.error(p,'Logistic'))
    #ann.predict_image("Test Images/two.png",p)
    ann.predict_test(299,p)









