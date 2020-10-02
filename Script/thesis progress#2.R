#load library
library(tsintermittent)
library(keras)
library(greybox)
library(DataCombine)

#data preparation
train <- read.csv('df_experiment2.csv', header=T, sep=',');head(train[,7:10])
test <- read.csv('df_experiment_test2.csv', header=T, sep=',');head(test[,7:10])
train_data <- data.frame(yt=c(t(train[3, 5:ncol(train)])))
test_data <- t(test[3, 7:ncol(test)])
train_data_binary <- data.frame(yt=c(ifelse(train_data$yt>1, 1, 0)))
test_data_binary <- ifelse(test_data>1, 1, 0)
#ACF check
acf(train_data$yt)
acf(diff(train_data$yt, differences = 1))

#croston model
crostonnaive <- crost(train_data, h=28, type='croston')
crostonsbj <- crost(train_data, h=28, type='sbj')
crostonsba <- crost(train_data, h=28, type='sba')

#deep learning
lag_transform <- function(train_data, lag=1){
  matrix_lag <- matrix(nrow=nrow(train_data), ncol=lag)
  colnames(matrix_lag) <- c(LETTERS[1:7])
  for(i in 1:lag){
    lagged <- DataCombine::slide(train_data, slideBy = -i, Var='yt', NewVar = 'a')$a
    matrix_lag[,i] <- lagged
    colnames(matrix_lag)[i] <- c(paste("yt_", as.character(i), sep=''))
  }
  df_lag <- data.frame(yt=c(train_data$yt), matrix_lag)
  return(df_lag)
}
train_keras <- lag_transform(train_data, 7)
train_keras <- train_keras[8:1913,]
train_keras <- (train_keras-min(train_data$yt))/(max(train_data$yt)-min(train_data$yt)) #0 1 scaling
test_keras <- cbind(yt_1=c(train_keras$yt[1906], rep(NA,27)), yt_2=c(train_keras$yt[1905:1906], rep(NA,26)),
                    yt_3=c(train_keras$yt[1904:1906], rep(NA,25)), yt_4=c(train_keras$yt[1903:1906], rep(NA,24)),
                    yt_5=c(train_keras$yt[1902:1906], rep(NA,23)), yt_6=c(train_keras$yt[1901:1906], rep(NA,22)),
                    yt_7=c(train_keras$yt[1900:1906], rep(NA,21)))
test_keras <- cbind(yt_7=c(train_keras$yt[1900:1906], rep(NA, 21)))
x_train <- as.matrix(train_keras[,2:8])
x_train <- as.matrix(train_keras[,2])
x_train_dnn <- as.matrix(train_keras[,2:8])
x_train_dnn <- as.matrix(train_keras[,2])
y_train <- as.matrix(train_keras$yt)
x_train_rnn <- array(x_train, dim=c(nrow(x_train), 1, ncol(x_train)))
set.seed(123)

#model dnn

model_dnn <- keras_model_sequential()
model_dnn %>%
  layer_dense(units=32, input_shape=3, activation='relu') %>%
  layer_dense(units=16, activation='relu') %>%
  layer_dense(units=1, activation='relu')

summary(model_dnn)

model_dnn %>%
  compile(
    loss='mse',
    optimizer='rmsprop',
    metrics='mae'
  )

train_dnn <- model_dnn %>%
  fit(
    x=x_train_dnn,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.015,
    shuffle=F
  )

#stacking croston model
train_stacking <- cbind(crostonnaive$frc.in, crostonsba$frc.in, crostonsbj$frc.in)
train_stacking <- train_stacking[c(sum(is.na(train_stacking[,1]))+1):nrow(train_stacking),]
train_stacking <- as.matrix(train_stacking)
y_train_stacking <- train_data$yt[c(sum(is.na(train_stacking[,1]))+1):nrow(train_stacking)]
train_stacking <- array(train_stacking, dim=c(nrow(train_stacking), 1, ncol(train_stacking))) #for recurrent model
test_stacking <- cbind(crostonnaive$frc.out, crostonsba$frc.out, crostonsbj$frc.out)
test_stacking <- array(test_stacking, dim=c(nrow(test_stacking), 1, ncol(test_stacking))) #for recurrent model

model_stacking <- keras_model_sequential()

model_stacking %>%
  layer_simple_rnn(input_shape=c(1,3), units=3) %>%
  layer_dense(1)

model_stacking %>%
  compile(
    loss='mae',
    optimizer='adam',
    metrics='mse'
  )

train_stacking <- model_stacking %>%
  fit(
    x=train_stacking,
    y=y_train_stacking,
    epochs=100,
    batch_size=32,
    validation_split=0.1
  )

f_stacking <- model_stacking %>% predict(test_stacking, batch_size=32)
f_stacking

#forecast deep learning recursive
train_keras
forecast_dnn <- function(te_keras){
  X_test <- te_keras
  pred <- numeric(28)
  for(k in 1:28){
    if(is.na(X_test$yt_1[k])){X_test$yt_1[k]=pred[k-1]}
    if(is.na(X_test$yt_2[k])){X_test$yt_2[k]=pred[k-2]}
    if(is.na(X_test$yt_3[k])){X_test$yt_3[k]=pred[k-3]}
    #if(is.na(X_test$yt_7[k])){X_test$yt_7[k]=pred[k-7]}
    X <- X_test[k,]
    X <- as.matrix(X)
    pred[k] <- model_dnn %>% predict(X, batch_size=32)
  }
  return(pred)
}
f_dnn <- forecast_dnn(as.data.frame(test_keras))
f_dnn <- (f_dnn*sd(train_data$yt))+mean(train_data$yt)
f_dnn_binary <- ifelse(f_dnn >0.5, 1, 0)
f_dnn_binary_reg <- f_dnn_binary*f_dnn

#model RNN
model_rnn <- keras_model_sequential()

model_rnn %>%
  layer_simple_rnn(input_shape = c(1,3), units=32, activation = 'sigmoid') %>%
  layer_dense(units=1, activation='sigmoid')

model_rnn %>%
  compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics='accuracy'
  )

train_rnn <- model_rnn %>%
  fit(
    x=x_train_rnn,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=F
  )

#forecast rnn
forecast_rnn <- function(te_keras){
  X_test <- te_keras
  pred <- numeric(28)
  for(k in 1:28){
    if(is.na(X_test$yt_1[k])){X_test$yt_1[k]=pred[k-1]}
    if(is.na(X_test$yt_2[k])){X_test$yt_2[k]=pred[k-2]}
    if(is.na(X_test$yt_3[k])){X_test$yt_3[k]=pred[k-3]}
    #if(is.na(X_test$yt_7[k])){X_test$yt_7[k]=pred[k-7]}
    X <- X_test[k,]
    X <- as.matrix(X)
    dim(X) <- c(1,1,3)
    pred[k] <- model_rnn %>% predict(X, batch_size=32)
  }
  return(pred)
}
f_rnn <- forecast_rnn(as.data.frame(test_keras))
f_rnn <- (f_rnn*sd(train_data$yt))+mean(train_data$yt)
1103+807
1-(1103/1910)
#model lstm
test_data <- data.frame(yt=c(t(test[1,7:ncol(test)])))
x_lag <- slide(test_data, slideBy=-c(1,2,3,4,5,6,7), Var='yt', NewVar=c('yt1','yt2','yt3','yt4','yt5','yt6','yt7'))
x_lag <- sapply(x_lag, function(x) ifelse(is.na(x), 0, x))
x_train <- as.data.frame(x_lag[1:1913,2:8])
x_test <- as.data.frame(x_lag[1914:1941,2:8])
x_train <- array(as.matrix(x_train), dim=c(nrow(x_train), 1, ncol(x_train)))
x_test <- array(as.matrix(x_test), dim=c(nrow(x_test), 1, ncol(x_test)))
y_train <- as.matrix(x_lag[1:1913,1])


model_lstm <- keras_model_sequential()

model_lstm %>%
  layer_lstm(input_shape=c(1,7), units=4) %>%
  layer_dense(units=1)

model_lstm %>%
  compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics='mae'
  )

for(i in 1:50){
  model_lstm %>% fit(x_train, y_train, epochs=1, batch_size=1, verbose=1, shuffle=F)
  model_lstm %>% reset_states()
}

train_lstm <- model_lstm %>%
  fit(
    x=x_train_rnn,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.015,
    shuffle=F,
    verbose=0
  )

#forecast lstm
k=1
forecast_lstm <- function(te_keras){
  X_test <- as.data.frame(te_keras)
  pred <- numeric(28)
  for(k in 1:28){
    if(is.na(X_test$yt_1[k])){X_test$yt_1[k]=pred[k-1]}
    if(is.na(X_test$yt_2[k])){X_test$yt_2[k]=pred[k-2]}
    if(is.na(X_test$yt_3[k])){X_test$yt_3[k]=pred[k-3]}
    if(is.na(X_test$yt_4[k])){X_test$yt_4[k]=pred[k-4]}
    if(is.na(X_test$yt_5[k])){X_test$yt_5[k]=pred[k-5]}
    if(is.na(X_test$yt_6[k])){X_test$yt_6[k]=pred[k-6]}
    if(is.na(X_test$yt_7[k])){X_test$yt_7[k]=pred[k-7]}
    X <- X_test[k,]
    X <- as.matrix(X)
    dim(X) <- c(1,1,7)
    pred[k] <- model_lstm %>% predict(X, batch_size=32)
  }
  return(pred)
}

f_lstm <- forecast_lstm(test_keras)
f_lstm <- f_lstm*(max(train_data$yt)-min(train_data$yt)) + min(train_data$yt)
f_lstm
pred <- model_lstm %>% predict(x_test, batch_size=1)


#model gru
model_gru <- keras_model_sequential()

model_gru %>%
  layer_gru(input_shape=list(NULL, 3), units=32) %>%
  layer_dense(units=1)

model_gru %>%
  compile(
    optimizer='rmsprop',
    loss='mse'
  )

train_gru <- model_gru %>%
  fit(
    x=x_train_rnn,
    y=y_train,
    epochs=100,
    validation_split=0.015,
    shuffle=F,
    batch_size=32
  )

#forecast gru
forecast_gru <- function(te_keras){
  X_test <- as.data.frame(te_keras)
  pred <- numeric(28)
  for(k in 1:28){
    if(is.na(X_test$yt_1[k])){X_test$yt_1[k]=pred[k-1]}
    if(is.na(X_test$yt_2[k])){X_test$yt_2[k]=pred[k-2]}
    if(is.na(X_test$yt_3[k])){X_test$yt_3[k]=pred[k-3]}
    #if(is.na(X_test$yt_7[k])){X_test$yt_7[k]=pred[k-7]}
    X <- X_test[k,]
    X <- as.matrix(X)
    dim(X) <- c(1,1,3)
    pred[k] <- model_gru %>% predict(X, batch_size=32)
  }
  return(pred)
}


f_gru <- forecast_gru(test_keras)
f_gru <- (f_gru*sd(train_data$yt))+mean(train_data$yt)

#evaluate model
eval_model <- function(te_data, tr_data, f_data){
  forecast_data <-c(tr_data, f_data)
  return(RMSSE(te_data, forecast_data, 28))
}
Metrics::rmse(test_data[1914:1941], crostonnaive$frc.out)
Metrics::rmse(test_data[1914:1941], crostonsba$frc.out)
Metrics::rmse(test_data[1914:1941], crostonsbj$frc.out)
eval_model(test_data, train_data$yt, crostonnaive$frc.out)
eval_model(test_data, train_data$yt, crostonsba$frc.out)
eval_model(test_data, train_data$yt, crostonsbj$frc.out)
eval_model(test_data, train_data$yt, f_dnn)
eval_model(test_data, train_data$yt, f_dnn_binary_reg)
eval_model(test_data, train_data$yt, f_rnn)
eval_model(test_data, train_data$yt, f_lstm)
eval_model(test_data, train_data$yt, f_gru)
RMSSE(test_data$yt, c(train_data$yt, pred), 28)
sum(test_data[1914:1941])
ts.plot(f_dnn)
sum(crostonnaive$frc.out)
sum(f_lstm)
sum(test_data[1914:1941])
Metrics::rmse(test_data[1914:1941], crostonnaive$frc.out)
Metrics::accuracy(test_data_binary[1914:1941], f_dnn_binary)
sum(crostonnaive$frc.out)
sum(test_data[1914:1941])
