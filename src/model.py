import tensorflow as tf 
from keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout
from keras.preprocessing.text import one_hot
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,one_hot
import pandas as pd 
import numpy as np
from config import *
from keras import Sequential
from keras.models import model_from_json



class lstm_model:
    def __init__(self,emb_size,data,fold,max_features,sent_len,voc_size):
        self.emb_size = emb_size
        self.data = data
        self.fold = fold
        self.max_features = max_features
        self.sent_len = sent_len
        self.voc_size = voc_size


    def preprocess_data(self):
            
        data = self.data  
        fold = self.fold
        max_features = self.max_features
        sent_len = self.sent_len

        df_train = data[ data.kfold != fold]
        df_test = data[data.kfold==fold]

        
        tokenizer = Tokenizer(num_words=max_features, split=' ')
            
        tokenizer.fit_on_texts(data['review'].values)

        X_train = tokenizer.texts_to_sequences(df_train['review'].values)
        X_train = pad_sequences(X_train,maxlen=sent_len)
        y_train = df_train['sentiment']

        X_test = tokenizer.texts_to_sequences(df_test['review'].values)
        X_test = pad_sequences(X_test,maxlen=sent_len)
        y_test = df_test['sentiment']
            
        return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

    def lstm(self):
        voc_size = self.voc_size
        emb_size = self.emb_size
        sent_len = self.sent_len
        model=Sequential()
        model.add(Embedding(voc_size,emb_size,input_length=sent_len))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dropout(0.3))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        return model

    def train_model(self):

        X_train,X_test,y_train,y_test = self.preprocess_data()
        
        model = self.lstm()
        print('training')
        model.fit(X_train, y_train, epochs = 7, batch_size=batch_size, verbose = 2,validation_data=[X_test,y_test])

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(r'model.h5')
        print("Saved model to disk")




if __name__ =='__main__':
    data = pd.read_csv(DATA_FILE)
    Model = lstm_model(emb_size,data,fold,max_features,sent_len,voc_size)
    Model.train_model()


    