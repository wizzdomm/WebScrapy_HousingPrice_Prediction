import pandas as pd
import numpy as np
from numpy.random import seed,randn
from numpy import mean,std
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import tkinter as Tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg

class prediction:


    def PredictionLoad(self):

        Form = Tk()
        Form.geometry('400x400')
        right = int(Form.winfo_screenwidth() / 2 - 400 / 2)
        down = int(Form.winfo_screenheight() / 2 - 400 / 2)
        Form.geometry('+{}+{}'.format(right, down))
        Form.resizable(0, 0)
        Form.title('Form')
        def check():

            data = pd.read_csv('dataset/HouseNew.csv')

            Elevator = float(entElevator.get())
            Floor = float(entFloor.get())
            Area = float(entArea.get())
            Parking = float(entParking.get())
            Room = float(entRoom.get())
            Warehouse = float(entWarehouse.get())
            Year = float(entYear.get())

            data.drop(['Address'],inplace= True,axis = 1)
            data['Floor'] = data['Floor'].fillna('median')
            data['Floor'] = data['Floor'].replace('median',3)
            data['Elevator'] = data['Elevator'].astype('float')
            data['Parking'] = data['Parking'].astype('float')
            data['Warehouse'] = data['Warehouse'].astype('float')
            seed(1)
            data_mean_area,data_std_area = mean(data['Area']),std(data['Area'])
            cut_off_area = 3*data_std_area * 3
            lower_area,upper_area = data_mean_area - cut_off_area,data_mean_area + cut_off_area
            outliersArea = [x for x in data['Area'] if x < lower_area or x > upper_area]
            # print(len(outliersArea))
            data['Area'] = pd.Series([x for x in data['Area'] if x >= lower_area and x <= upper_area])

            # print(data.isnull().sum())
            # print(data['Area'].mean())
            data['Area'] = data['Area'].fillna(98.55)
            # print(data.isnull().sum())
            X = data.drop(['Price'],axis = 1)
            y = data['Price']

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=34)
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            model = LinearRegression()
            model.fit(X_train,y_train)
            y_hat = model.predict(X_test)
            # print(y_hat)

            MAE = mean_absolute_error(y_test,y_hat)
            MSE = mean_squared_error(y_test,y_hat)
            RMSE = np.sqrt(MSE)
            # print('MAE: ',MAE)
            # print()
            # print('MSE: ',MSE)
            # print()
            # print('RMSE: ',RMSE)
            #
            # print(data['Price'].mean())
            # print(data['Price'].mean()/RMSE)

            test_residual = y_test- y_hat
            sns.scatterplot(x = y_test,y= test_residual)
            plt.axhline(y = 0,color = 'r',linestyle = '--')
            # plt.show()
            # new_data = [[1.0,1.0,311,1.0,4,1.0,1396]]
            # print(model.predict(new_data))
            result = model.predict([[Elevator,Floor,Area,Parking,Warehouse,Year,Room]])
            msg.showinfo('Prediction', str(result))

        lblElevator = ttk.Label(Form,text='Elavator: ')
        lblElevator.grid(row=0,column=0,padx=10,pady=10)
        elavator = [0,1]
        entElevator = ttk.Combobox(Form,values=elavator,width=20)
        entElevator.grid(row=0,column=1,padx=10,pady=10)

        lblFloor = ttk.Label(Form,text='Floor: ')
        lblFloor.grid(row=1,column=0,pady=10,padx=10)

        entFloor = ttk.Entry(Form,width=23)
        entFloor.grid(row=1,column=1,padx=10,pady=10)

        lblArea = ttk.Label(Form,text='Area: ')
        lblArea.grid(row=2,column=0,pady=10,padx=10)

        entArea = ttk.Entry(Form,width=23)
        entArea.grid(row=2,column=1,padx=10,pady=10)

        lblParking = ttk.Label(Form,text='Parking: ')
        lblParking.grid(row=3,column=0,padx=10,pady=10)
        Parking = [0,1]
        entParking = ttk.Combobox(Form,values=Parking,width=20)
        entParking.grid(row=3,column=1,padx=10,pady=10)

        lblRoom = ttk.Label(Form,text='Room: ')
        lblRoom.grid(row=4,column=0,pady=10,padx=10)

        entRoom = ttk.Entry(Form,width=23)
        entRoom.grid(row=4,column=1,padx=10,pady=10)


        lblWarehouse = ttk.Label(Form,text='Warehouse: ')
        lblWarehouse.grid(row=5,column=0,padx=10,pady=10)
        Warehouse = [0,1]
        entWarehouse = ttk.Combobox(Form,values=Warehouse,width=20)
        entWarehouse.grid(row=5,column=1,padx=10,pady=10)

        lblYear = ttk.Label(Form,text='Year: ')
        lblYear.grid(row=6,column=0,pady=10,padx=10)

        entYear = ttk.Entry(Form,width=23)
        entYear.grid(row=6,column=1,padx=10,pady=10)

        btnPredict = ttk.Button(Form,text='Predict',width=13,command= check)
        btnPredict.grid(row=7,column=3,padx=10,pady=10,sticky='e')

        Form.mainloop()