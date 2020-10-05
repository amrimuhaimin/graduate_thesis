#load library
library(tsintermittent)
library(keras)
library(greybox)
library(DataCombine)

setwd("~/graduate_thesis/Data/")
#data preparation
train <- read.csv('df_experiment2.csv', header=T, sep=',');head(train[,5:10])
test <- read.csv('df_experiment_test2.csv', header=T, sep=',');head(test[,7:10])
train_data <- data.frame(yt=c(t(train[1, 5:ncol(train)])))
test_data <- data.frame(yt=c(t(test[1, 7:ncol(test)])))
lag <- 28
batch_size <- 32
verbose <- 0
epochs <- 100
shuffle <- FALSE
val_split <- 0.2
features <- 1
gru_units <- 4
dense_units <- 1
rec_dropout <- 0.1
dropout <- 0.1
out_act <- 'sigmoid'
loss <- 'mean_squared_error'
optimizer <- 'adam'
metrics <- 'mae'

######non-recursive gru model 1 time steps and n features############
model.gru_nonrc <- function(train_data, test_data, features=1, lag=7, batch_size=32, verbose=0,
                      epochs=100, shuffle=F, val_split=0.015, loss='mean_squared_error',
                      optimizer='adam', metrics='mae', gru_units=4, dense_units=1, dropout=0,
                      rec_dropout=0, out_act='sigmoid'){
  
  #preprocess
  lag_transform <- function(train_data, lag=1){
    matrix_lag <- matrix(nrow=nrow(train_data), ncol=lag)
    colnames(matrix_lag) <- c(LETTERS[1:lag])
    for(i in lag:1){
      lagged <- DataCombine::slide(train_data, slideBy = -i, Var='yt', NewVar = 'a')$a
      matrix_lag[,lag-(i-1)] <- lagged
      colnames(matrix_lag)[lag-(i-1)] <- c(paste("yt_", as.character(i), sep=''))
    }
    df_lag <- data.frame(yt=c(train_data$yt), matrix_lag)
    return(df_lag)
  }
  
  data_trans <- lag_transform(test_data, lag)
  data_trans <- data_trans[(lag+1):nrow(data_trans),]
  data_trans <- (data_trans-min(train_data$yt))/(max(train_data$yt)-min(train_data$yt)) #0 1 scaling
  train_keras <- data_trans[1:(nrow(data_trans)-28),]
  test_keras <- data_trans[-(1:(nrow(data_trans)-28)),]
  
  x_train <- as.matrix(train_keras[,2:(lag+1)])
  y_train <- as.matrix(train_keras$yt)
  x_test <- as.matrix(test_keras[,2:(lag+1)])
  x_train_rnn <- array(x_train, dim=c(nrow(x_train), 1, ncol(x_train)))  
  x_test_rnn <- array(x_test, dim=c(nrow(x_test), 1, ncol(x_test)))
  
  #if(features > 1){
  #  x_train_rnn <- array(x_train, dim=c(nrow(x_train), features, ncol(x_train)))  
  #} else {
  #  x_train_rnn <- array(x_train, dim=c(nrow(x_train), ncol(x_train), features))
  #}
  
  set.seed(123)
  
  model_gru <- keras_model_sequential()
  
  model_gru %>%
    layer_gru(input_shape=c(dim(x_train_rnn)[2], dim(x_train_rnn)[3]), units=gru_units, activation = "linear",
              recurrent_activation = "linear", dropout=dropout, recurrent_dropout=rec_dropout ) %>%
    layer_dense(units=dense_units, activation = out_act)
  
  model_gru %>%
    compile(
      optimizer='adam',
      loss='mean_squared_error',
      metrics='mae'
    )
  
  train_gru <- model_gru %>%
    fit(
      x=x_train_rnn,
      y=y_train,
      epochs=epochs,
      validation_split=val_split,
      shuffle=shuffle,
      batch_size=batch_size,
      verbose=verbose
    )
  
  f_gru <- model_gru %>% predict(x_test_rnn, batch_size=batch_size)
  f_gru_test <- f_gru*(max(train_data$yt)-min(train_data$yt)) + min(train_data$yt)
  f_gru_train <- model_gru %>% predict(x_train_rnn, batch_size=batch_size)
  f_gru_train <- f_gru_train*(max(train_data$yt)-min(train_data$yt)) + min(train_data$yt)
  
  return(list(f_gru_test, f_gru_train))
}

f_gru <- model.gru_nonrc(train_data=train_data, test_data = test_data, lag=28, batch_size=32,
                   val_split=NULL, shuffle=F, optimizer='adam', gru_units=128, rec_dropout=0.1, dropout=0.1,
                   epochs=100, out_act='sigmoid')

#evaluate model
eval_model <- function(te_data, tr_data, f_data, h){
  forecast_data <-c(tr_data, f_data)
  return(RMSSE(te_data, forecast_data, h))
}

eval_model(test_data$yt, train_data$yt, f_gru[[1]], 28)

#parameter tune
optim <- c('rmsprop', 'adam')
gu <- c(4, 32, 64, 128)
rd <- c(0.1, 0.2, 0.3)
do <- c(0.1, 0.2, 0.3)
ep <- c(100, 300, 500)


i <- 1
result.list <- list()
for(g in gu){
  for(r in rd){
    for(d in do){
      for(e in ep){
        for(o in optim){
          f_gru <- model.gru_nonrc(train_data=train_data, test_data = test_data, lag=28, batch_size=32,
                                   val_split=NULL, shuffle=F, optimizer=o, gru_units=g, rec_dropout=r, dropout=d,
                                   epochs=e, out_act='linear')
          i <- i+1
          result.list[[i]] <- f_gru
        }
      }
    }
  }
}
plot(test_data$yt[1914:1941], type='l', col='blue')
lines(f_gru, col='red')
sum(f_gru)
sum(test_data$yt[1914:1941])

