library(tsintermittent)
library(keras)
library(greybox)

setwd("E:/NCTU/Thesis Ref/Data/M5/")
dt=1
lag=28
h=28
#data preparation
train_data_sim2 <- data.frame(yt=c(sim2$ts.1[1:(1825-h)]))
train <- read.csv('df_experiment2.csv', header=T, sep=',');head(train[,5:10])
test <- read.csv('df_experiment_test2.csv', header=T, sep=',');head(test[,7:10])
train_data <- data.frame(yt=c(t(train[dt, 5:ncol(train)])))
train_data_demand <- as.data.frame(crost.decomp(data.frame(yt=c(t(train[dt, 5:ncol(train)]))), "naive")[1]); colnames(train_data_demand) <- 'yt'
train_data_interval <- as.data.frame(crost.decomp(data.frame(yt=c(t(train[dt, 5:ncol(train)]))), 'naive')[2]); colnames(train_data_interval) <- 'yt'
test_data <- data.frame(yt=c(t(test[dt, 7:ncol(test)])))
train_data_demand <- as.data.frame(crost.decomp(data.frame(yt=c(t(train_data_sim2))), "naive")[1]); colnames(train_data_demand) <- 'yt'
train_data_interval <- as.data.frame(crost.decomp(data.frame(yt=c(t(train_data_sim2))), 'naive')[2]); colnames(train_data_interval) <- 'yt'


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

lag_transform2 <- function(trn_data, lag=1, h=28){
  r <- nrow(trn_data)
  matrix_lag <- matrix(nrow=h, ncol=lag)
  colnames(matrix_lag) <- c(LETTERS[1:lag])
  for(i in lag:1){
    lagged <- c(trn_data$yt[(r-i+1):r], rep(NA, h-i))
    matrix_lag[,lag-(i-1)] <- lagged
    colnames(matrix_lag)[lag-(i-1)] <- c(paste("yt_", as.character(i), sep=''))
  }
  df_lag <- data.frame(matrix_lag)
  return(df_lag)
}

train_keras_demand <- lag_transform(train_data_demand, lag=lag); train_keras_demand <- train_keras_demand[(1+lag):nrow(train_keras_demand),]
train_keras_interval <- lag_transform(train_data_interval, lag=lag); train_keras_interval <- train_keras_interval[(1+lag):nrow(train_keras_interval),]
train_keras_demand <- (train_keras_demand-min(train_data_demand$yt))/(max(train_data_demand$yt)-min(train_data_demand$yt))
train_keras_interval <- (train_keras_interval-min(train_data_interval$yt))/(max(train_data_interval$yt)-min(train_data_interval$yt))
x_train_demand <- as.matrix(train_keras_demand[1:(nrow(train_keras_demand)-h),-1])
x_train_interval <- as.matrix(train_keras_interval[1:(nrow(train_keras_interval)-h),-1])
y_train_demand <- as.matrix(train_keras_demand[1:(nrow(train_keras_demand)-h),1])
y_train_interval <- as.matrix(train_keras_interval[1:(nrow(train_keras_interval)-h),1])
test_keras_demand <- lag_transform2(train_keras_demand, lag, h)
test_keras_interval <- lag_transform2(train_keras_interval, lag, h)

#reshape training set.
x_train_abind <- abind::abind(array(x_train_demand, dim=c(nrow(x_train_demand), ncol(x_train_demand), 1)),
                              array(x_train_interval, dim=c(nrow(x_train_demand), ncol(x_train_demand), 1)))
y_train_abind <- abind::abind(array(y_train_demand, dim=c(nrow(y_train_demand), ncol(y_train_demand), 1)),
                                    array(y_train_interval, dim=c(nrow(y_train_demand), ncol(y_train_demand), 1)))
test_keras_abind <- abind::abind(array(as.matrix(test_keras_demand), dim=c(nrow(test_keras_demand), ncol(test_keras_demand), 1)),
                     array(as.matrix(test_keras_interval), dim=c(nrow(test_keras_demand), ncol(test_keras_demand), 1)))
  
in_dim <- c(dim(x_train_abind)[2], dim(x_train_abind)[3])
out_dim <- dim(y_train_abind)[3]

model_molstm <- keras_model_sequential() %>%
  layer_lstm(units=128, input_shape=c(in_dim), recurrent_activation="linear", activation="linear", dropout=0.1,
             recurrent_dropout = 0.1) %>%
  layer_dense(units=out_dim, activation='sigmoid') %>%
  compile(metrics="mae",
    loss="mean_squared_error",
    optimizer="adam")
history <- model_molstm %>%
  fit(x=x_train_abind,
      y=y_train_abind,
      epochs=100,
      validation_split=NULL,
      shuffle=F,
      batch_size=32,
      verbose=0)

forecast_molstm <- function(h){
  Xdemand <- test_keras_demand
  Xinterval <- test_keras_interval
  #Xtest <- test_keras_abind
  pred <- matrix(ncol=out_dim, nrow=h)
  for (i in 1:h){
    for (l in lag:1){
      if(is.na(Xdemand[i,l])){Xdemand[i,l]=pred[i-(lag-(l-1)),1]}
      if(is.na(Xinterval[i,l])){Xinterval[i,l]=pred[i-(lag-(l-1)),2]}
    }
    X <- abind::abind(array(as.matrix(Xdemand[i,]),dim=c(1,lag,1)),
                      array(as.matrix(Xinterval[i,]),dim=c(1,lag,1)))
    pred[i,] <- model_molstm %>% predict(X, batch_size=32)
  }
  return(pred)
}

f_molstm <- forecast_molstm(h)
f_molstm[,1] <- (f_molstm[,1]*(max(train_data_demand$yt)-min(train_data_demand$yt)))+min(train_data_demand)
f_molstm[,2] <- (f_molstm[,2]*(max(train_data_interval$yt)-min(train_data_interval$yt)))+min(train_data_interval)


eval_model <- function(te_data, tr_data, f_data, h){
  forecast_data <-c(tr_data, f_data)
  return(RMSSE(te_data, forecast_data, h))
}

eval_model(test_data_sim$yt, train_data_sim$yt, f_molstm[,1])
cumsum(round(f_molstm[,2]))
sum(f_molstm[1:14,1])

sum(test_data$yt[1914:1941])
undecompose <- function(interdemand, demand, h){
  k <- 0
  i <- 1
  v.value <- numeric(0)
  while(k < h){
    value <- c(rep(0, floor(interdemand[i])-1), demand[i])
    v.value <- c(v.value, value)
    k <- k + floor(interdemand[i])
    i <- i+1
  }
  v.value <- v.value[1:h]
  return(v.value)
}

f_rmolstm <- undecompose(f_molstm[,2], f_molstm[,1], h);f_rmolstm
eval_model(test_data_sim$yt, train_data_sim2$yt, f_rmolstm, h)
sum(tail(test_data_sim$yt,h))
abs(sum(f_rmolstm)-sum(tail(test_data_sim$yt,h)))

#test123