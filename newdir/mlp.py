import silence_tensorflow #per aver pronto un package per silenziare i warning di tensorflow (e non comandi)
import matplotlib #lib grafici
matplotlib.use("Agg") #backend agg e non grafico perchè si vuole usare in sistemi privi si GUI come travis
import matplotlib.pyplot as plt #importiamo renderer pyplot
import extra_keras_metrics #package per usare metriche come auPRC, auROC, ecc. permette di usare metrics="auroc" direttamente. rende disponibile ttutte le metriche di tensorflow in keras. Metriche su regressione: non servono auROC, auPRC
from extra_keras_utils import set_seed #funzione vista in formato compresso #if gpu available, per sapere se keras sui è avviato 
from keras.datasets import boston_housing #keras offre dataset standard per ottimizzare la rete (per rendere reti confrontabili)
from keras.models import Sequential, load_model   #modello standard è il sequential (altri modelli, ma questo ok nel 99% casi. altri pvanno bene per reti ad albero)
from keras.layers import Dense, InputLayer, Dropout #percettrone multilivello
from plot_keras_history import plot_history #altro package per plottare la history
import pandas as pd  #per dataframe e funzioni utili
import os
import numpy as np

def mlp(epochs:int):   #definiamo funz mlp che riceve il solo parametro: numero epoch su cui allenare il modello); set_seed accetta anche il parametro kill_parallelism che permette di evitare che l'ordine dei processi 8che non è deterministico) condizioni la randomizzazione. kill fa andare su un solo thread ma sarà lento
    set_seed(42)  
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    #print(x_train.shape) #per sapere come è fatto
    model = Sequential([
        InputLayer(input_shape=(x_train.shape[1],)),
        Dense(150, activation="relu"),
        Dense(150, activation= "relu"),
        Dense(150, activation= "relu"),
        Dense(150, activation= "relu"),
        Dense(150, activation= "relu"),
        Dropout(0.2),
        Dense (1, activation= "relu") #un solo neurone di output: ha valori di float da 0 a 1 e sopra peso che può assumere tutti i valori che si vuole, ma sigmoide fa ottenere valori 0 o 1.
    ])
    #model compile a cui si passano optimizer, loss e metriche

    model.compile(
        optimizer="nadam",
        loss="mse" #va bene questa perchè le altre metriche risultano sfalsate essendo una regressione
        )

    #fit al modello

    history = model.fit(
        x_train, y_train, #da dataset
        validation_data= (x_test, y_test), #da dataset
        epochs= epochs,
        batch_size= 100,
        shuffle= True #permette di rimischiare a ogni epochs i dati di training
        ).history
    #dopo fit: se modello e fare andare con tanti pesi weight = model.get_weights(); model.save....


    model.save("model.h5") #salva modello allenato

    pd.DataFrame(history).to_csv("history.csv") #per salvare history
    
    #plot_history(history)

    plt.savefig("history.png")



#fare andare le predizioni usando un modello già trainato

def predict(x:np.ndarray):
    cwd = os.path.dirname(os.path.realpath(__file__))
    model = load_model("{cwd}/model.h5".format(cwd=cwd))
    return model.predict(x)
