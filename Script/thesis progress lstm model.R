#load library
library(tsintermittent)
library(keras)
library(greybox)
library(DataCombine)

setwd("E:/NCTU/Thesis Ref/Data/M5/")
#data preparation
train <- read.csv('df_experiment2.csv', header=T, sep=',');head(train[,5:10])
test <- read.csv('df_experiment_test2.csv', header=T, sep=',');head(test[,7:10])
train_data <- data.frame(yt=c(t(train[3, 5:ncol(train)])))
test_data <- data.frame(yt=c(t(test[3, 7:ncol(test)])))
lag <- 28
batch_size <- 32
verbose <- 0
epochs <- 100
shuffle <- FALSE
val_split <- 0.015
features <- 1
lstm_units <- 32
dense_units <- 1
dropout <- 0
rec_dropout <- 0
out_act <- 'sigmoid'
loss <- 'mean_squared_error'
optimizer <- 'adam'
metrics <- 'mae'


######lstm model 1 time steps and n features############
model.lstm <- function(train_data, test_data, features=1, lag=7, batch_size=32, verbose=0,
                       epochs=100, shuffle=F, val_split=0.015, loss='mean_squared_error',
                       optimizer='adam', metrics='mae', lstm_units=4, dense_units=1, dropout=0, rec_dropout=0,
                       out_act='sigmoid'){
  
  #preprocess
  lag_transform <- function(train_data, lag=1){
    matrix_lag <- matrix(nrow=nrow(train_data), ncol=lag)
    colnames(matrix_lag) <- c(LETTERS[1:lag])
    for(i in 1:lag){
      lagged <- DataCombine::slide(train_data, slideBy = -i, Var='yt', NewVar = 'a')$a
      matrix_lag[,i] <- lagged
      colnames(matrix_lag)[i] <- c(paste("yt_", as.character(i), sep=''))
    }
    df_lag <- data.frame(yt=c(train_data$yt), matrix_lag)
    #df_lag <- df_lag[,order(names(df_lag), decreasing=T)]
    return(df_lag)
  }
  
  train_keras <- lag_transform(train_data, lag)
  train_keras <- train_keras[(lag+1):nrow(train_keras),]
  train_keras <- (train_keras-min(train_data$yt))/(max(train_data$yt)-min(train_data$yt)) #0 1 scaling
  
  lag_transform2 <- function(trn_data, lag=1, h=28){
    r <- nrow(trn_data)
    matrix_lag <- matrix(nrow=h, ncol=lag)
    colnames(matrix_lag) <- c(LETTERS[1:lag])
    for(i in 1:lag){
      lagged <- c(trn_data$yt[(r-i+1):r], rep(NA, h-i))
      matrix_lag[,i] <- lagged
      colnames(matrix_lag)[i] <- c(paste("yt_", as.character(i), sep=''))
    }
    df_lag <- data.frame(matrix_lag)
    #df_lag <- df_lag[,order(names(df_lag), decreasing=T)]
    return(df_lag)
  }
  
  test_keras <- lag_transform2(train_keras, lag)
  
  x_train <- as.matrix(train_keras[,2:(lag+1)])
  y_train <- as.matrix(train_keras$yt)
  x_train_rnn <- array(x_train, dim=c(nrow(x_train), features, ncol(x_train)))
  
  set.seed(123)
  
  #modeling lstm
  model_lstm <- keras_model_sequential()
  
  model_lstm %>%
    layer_lstm(input_shape=c(dim(x_train_rnn)[2], dim(x_train_rnn)[3]), units=lstm_units, activation='linear',
               recurrent_activation='linear', dropout=dropout, recurrent_dropout=rec_dropout) %>%
    layer_dense(units=dense_units, activation=out_act)
  
  summary(model_lstm)
  
  model_lstm %>%
    compile(
      loss=loss,
      optimizer=optimizer,
      metrics=metrics
    )
  
  train_lstm <- model_lstm %>%
    fit(
      x=x_train_rnn,
      y=y_train,
      epochs=epochs,
      batch_size=batch_size,
      validation_split=val_split,
      shuffle=shuffle,
      verbose=verbose
    )
  te_keras <- test_keras
  k=2
  #forecast lstm recursive
  forecast_lstm <- function(te_keras){
    X_test <- as.data.frame(te_keras)
    pred <- numeric(28)
    for(k in 1:28){
      for(l in 1:lag){
        if(is.na(X_test[k,l])){X_test[k,l]=pred[k-l]}  
      }
      X <- as.data.frame(X_test[1:k,])
      X <- as.matrix(X)
      dim(X) <- c(k,dim(x_train_rnn)[2],dim(x_train_rnn)[3])
      value <- model_lstm %>% predict(X, batch_size=batch_size)
      pred[k] <- value[k]
    }
    return(pred)
  }
  
  f_lstm <- forecast_lstm(test_keras)
  f_lstm_test <- f_lstm*(max(train_data$yt)-min(train_data$yt))+min(train_data$yt)
  f_lstm_train <- model_lstm %>% predict(x_train_rnn)
  f_lstm_train <- f_lstm_train*(max(train_data$yt)-min(train_data$yt))+min(train_data$yt)
  return(list(f_lstm_test, f_lstm_train))
}

f_lstm <- model.lstm(train_data, test_data, lag=28, batch_size=32, optimizer = 'adam', lstm_units = 128,
                     epochs=100, val_split=NULL, shuffle=F, dropout = 0.2, rec_dropout = 0.2,
                     out_act = 'sigmoid')

#evaluate model
eval_model <- function(te_data, tr_data, f_data){
  forecast_data <-c(tr_data, f_data)
  return(RMSSE(te_data, forecast_data, 28))
}

eval_model(test_data$yt, train_data$yt, f_lstm[[1]], 28)

plot(test_data$yt[1914:1941], type='l', col='red')
lines(f_lstm[[1]], col='blue')
