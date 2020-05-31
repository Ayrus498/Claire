import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os


def FolderCheck(dirName):
	if not os.path.exists(dirName):
		 os.mkdir(dirName)

def HyperParameters():
	######### Hyper Parameters ##############
	Data_File="16_Mar_Dataset_3"
	Iteration_Value=10000
	test_Iteration=1000

#########################################
Generic_Folder="%d-%m-%Y_at_%H-%M-%S_"+Data_File
#########################################
FolderCheck('Data')
FolderCheck('Training_Log')
FolderCheck('Train_Test')
FolderCheck('Model')
FolderCheck('Graphs')
########## ANN Framework Structure ###################
now = datetime.now()
=======
now = datetime.now()

######### Hyper Parameters ##############
Data_File="16_Mar_Dataset_3"
Iteration_Value=10000
test_Iteration=1000
#########################################
Generic_Folder="%d-%m-%Y_at_%H-%M-%S_"+Data_File
########## ANN Framework Structure ###################
Source_Data="Data\\"+Data_File+".csv"
Log_File = now.strftime("Training_Log\\"+Generic_Folder+"_Training_Log.csv")
Train_Test=now.strftime("Train_Test\\"+Generic_Folder)
os.mkdir(Train_Test)
Model_Folder=now.strftime("Model\\"+Generic_Folder)
os.mkdir(Model_Folder)
Actual_Output=Data_File+"_ANN_Output.csv"
Graph=now.strftime("Graphs\\"+Generic_Folder)
os.mkdir(Graph)
######################################################
global TP_Class_0_Cal_Accuracy
global TP_Class_1_Cal_Accuracy
TP_Class_0_Cal_Accuracy=0
TP_Class_1_Cal_Accuracy=0


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        data=pd.read_csv(Source_Data)
        Col=data.shape[1]-1
        del data
        self.synaptic_weights = 2 *np.random.random((int(Col),1))-1

    def DataSplit(self):
        data=pd.read_csv(Source_Data)
        y=data.Output
        x=data.drop('Output',axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        x_train.to_csv(Train_Test+"\\Train_data.csv", index = False, header=False)
        x_test.to_csv(Train_Test+"\\Test_data.csv", index = False, header=False)
        y_train.to_csv(Train_Test+"\\Train_data_label.csv", index = False, header=False)
        y_test.to_csv(Train_Test+"\\Test_data_label.csv", index = False, header=False)

        Train_data = genfromtxt(Train_Test+"\\Train_data.csv", delimiter=',')
        Test_data = genfromtxt(Train_Test+"\\Test_data.csv", delimiter=',')
        Train_data_Label = genfromtxt(Train_Test+"\\Train_data_label.csv", delimiter=',')
        Test_data_Label = genfromtxt(Train_Test+"\\Test_data_label.csv", delimiter=',')

        return Train_data,Train_data_Label,Test_data,Test_data_Label

    def genrate_model(self, Model_File_Path, synaptic_weights):
        #synaptic_weights.to_csv(Model_File_Path, index = False, header=False)
        pd.DataFrame(synaptic_weights).to_csv(Model_File_Path,index = False, header=False)

    def Plot(self):
        x = []
        y = []
        z = []
        with open(Log_File,'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(float(row[0]))
                y.append(float(row[1]))
                z.append(float(row[2])) 
        # Error Graph
        plt.figure(0)
        plt.plot(x,y, label='Training Flow')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Iteration VS Error')
        plt.legend()
        plt.savefig(Graph+'\\Iteration_VS_Error.png')
        # Accuracy graph
        plt.figure(1)
        plt.plot(x,z, label='Accuracy Flow')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy %')
        plt.title('Iteration VS Accuracy')
        plt.legend()
        plt.savefig(Graph+'\\Iteration_VS_Accuracy.png')
        plt.show()


    def Log(self,iteration,error,accuracy):
        f = open(Log_File, "a")
        data=str(iteration)+","+str(error)+","+str(accuracy)
        f.write(data)
        f.write("\n")
        f.close()

    def sigmoid(self,x):
        return 1 / (1+ np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1-x)

    def test_accuracy(self,test_inputs,test_outputs):
        global TP_Class_0_Cal_Accuracy
        global TP_Class_1_Cal_Accuracy
        global FN_Class_0_Cal_Accuracy
        global FN_Class_1_Cal_Accuracy
        TP_Class_0_Cal_Accuracy=0
        TP_Class_1_Cal_Accuracy=0
        FN_Class_0_Cal_Accuracy=0
        FN_Class_1_Cal_Accuracy=0

        test_inputs = test_inputs.astype(float)
        Expected_outputs = test_outputs.astype(int)
        Actual_output = self.sigmoid(np.dot(test_inputs, self.synaptic_weights))
        loop=Expected_outputs.shape[0]
        i=0
        for i in range(loop):
            if(Actual_output[i] >0.5):
                Actual_output[i]=1
            else:
                Actual_output[i]=0

        Actual_output=Actual_output.astype(int)
        pd.DataFrame(Actual_output).to_csv(Train_Test+"\\"+Actual_Output,index = False, header=False)
        for i in range(loop):
            if(Expected_outputs[i]==0):
                if(Expected_outputs[i]==Actual_output[i]):
                    TP_Class_0_Cal_Accuracy=TP_Class_0_Cal_Accuracy+1
                else:
                    FN_Class_0_Cal_Accuracy=FN_Class_0_Cal_Accuracy+1
            else:
                if(Expected_outputs[i]==Actual_output[i]):
                    TP_Class_1_Cal_Accuracy=TP_Class_1_Cal_Accuracy+1
                else:
                    FN_Class_1_Cal_Accuracy=FN_Class_1_Cal_Accuracy+1

        TP_Class_0_Cal_Accuracy=(TP_Class_0_Cal_Accuracy/loop)*100
        TP_Class_1_Cal_Accuracy=(TP_Class_1_Cal_Accuracy/loop)*100
        FN_Class_0_Cal_Accuracy=(FN_Class_0_Cal_Accuracy/loop)*100
        FN_Class_1_Cal_Accuracy=(FN_Class_1_Cal_Accuracy/loop)*100
        
        return TP_Class_0_Cal_Accuracy,FN_Class_0_Cal_Accuracy,TP_Class_1_Cal_Accuracy,FN_Class_1_Cal_Accuracy

    def train(self, training_inputs, training_outputs, test_inputs, test_outputs, training_iterations):
        global test_Iteration
        tracker=test_Iteration
        Accuracy_Test=0
        global TP_Class_0_Accuracy_Test
        global TP_Class_1_Accuracy_Test
        global FN_Class_0_Accuracy_Test
        global FN_Class_1_Accuracy_Test
        TP_Class_0_Accuracy_Test=0
        TP_Class_1_Accuracy_Test=0
        FN_Class_0_Accuracy_Test=0
        FN_Class_1_Accuracy_Test=0
        for iterations in range(training_iterations):               
            if(iterations==test_Iteration):
                print("[INFO] Reached ",iterations," iteration of Training Process")
                print("[AverageError] Calculated Error",avgerror)
                print("[Accuracy] Calculated Accuracy for Class 0",TP_Class_0_Accuracy_Test)
                print("[Accuracy] Calculated Accuracy for Class 1",TP_Class_1_Accuracy_Test)
                print("\n")
                Model_File_Path=now.strftime(Model_Folder+"\\"+Generic_Folder+str(iterations)+"_weights.model")
                self.genrate_model(Model_File_Path,self.synaptic_weights)
                #Accuracy_Test=self.test_accuracy(test_inputs,test_outputs)
                TP_Class_0_Accuracy_Test,FN_Class_0_Accuracy_Test,TP_Class_1_Accuracy_Test,FN_Class_1_Accuracy_Test=self.test_accuracy(test_inputs,test_outputs)
                Accuracy_Test=TP_Class_0_Accuracy_Test+TP_Class_1_Accuracy_Test
                Accuracy_Test=str(Accuracy_Test)+","+str(TP_Class_0_Accuracy_Test)+","+str(FN_Class_0_Accuracy_Test)+","+str(TP_Class_1_Accuracy_Test)+","+str(FN_Class_1_Accuracy_Test)
                self.Log(iterations,avgerror,Accuracy_Test)
                test_Iteration+=tracker
                
            output = self.think(training_inputs)
            error = training_outputs - output
            avgerror=abs(np.average(error))
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__=="__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weigts:")
    print(neural_network.synaptic_weights)
    
    training_inputs,training_outputs,test_inputs,test_outputs=neural_network.DataSplit()
    training_outputs=np.array([training_outputs]).T

    neural_network.train(training_inputs, training_outputs, test_inputs, test_outputs, Iteration_Value)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    neural_network.Plot()

