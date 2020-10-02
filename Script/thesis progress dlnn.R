#load library
library(vroom)
library(neuralnet)

path <- "E:/NCTU/Thesis Ref/Data/M5/"
calendar <- readr::read_csv(stringr::str_c(path, 'calendar.csv'), col_types = cols())
dummy_event1 <- ifelse(is.na(calendar$event_name_1)==TRUE,0,1)
dummy_weekend <- as.data.frame(fastDummies::dummy_cols(calendar, "weekday"))
dummy_weekend <- dummy_weekend[,15:21]
dummy_snap <- as.data.frame(calendar[,12:14])
df_dummy <- data.frame(dummy_event1, dummy_weekend, dummy_snap);head(df_dummy)
N <- 30490
sample_sub <- matrix(nrow=28,ncol=5)
set.seed(123)
#load data frame
df <- vroom(stringr::str_c(path,'sales_train_validation.csv'), delim=",", col_types = cols())
evaluation <- vroom(stringr::str_c(path, 'sales_train_evaluation.csv'), delim=',', col_types= cols())
run_dlnn <- function(){
  path <- "E:/NCTU/Thesis Ref/Data/M5/"
  calendar <- readr::read_csv(stringr::str_c(path, 'calendar.csv'), col_types = cols())
  dummy_event1 <- ifelse(is.na(calendar$event_name_1)==TRUE,0,1)
  dummy_weekend <- as.data.frame(fastDummies::dummy_cols(calendar, "weekday"))
  dummy_weekend <- dummy_weekend[,15:21]
  dummy_snap <- as.data.frame(calendar[,12:14])
  df_dummy <- data.frame(dummy_event1, dummy_weekend, dummy_snap);head(df_dummy)
  N <- 30490
  sample_sub <- matrix(nrow=28,ncol=N)
  set.seed(123)
  #load data frame
  df <- vroom(stringr::str_c(path,'sales_train_validation.csv'), delim=",", col_types = cols())
  evaluation <- vroom(stringr::str_c(path, 'sales_train_evaluation.csv'), delim=',', col_types= cols())
  for(i in 1:N){
    svMisc::progress(i, N)
    #preprocess data
    train <- data.frame(df_dummy[1:1913,], yt=c(t(df[5680,(7:ncol(df))])))
    test <- data.frame(df_dummy[1914:1941,], yt=c(t(evaluation[5680,(1914:1941)])))
    #remove(df,evaluation)
    train_min_max <- train
    train_min_max$yt <- train$yt/max(train$yt)
    train_min_max <- rbind(train_min_max[,order(names(train))], test[,order(names(test))])
    yt_lag1 <- DataCombine::slide(train_min_max, slideBy=-1, Var='yt', NewVar='yt_lag1')$yt_lag1
    yt_lag7 <- DataCombine::slide(train_min_max, slideBy=-7, Var='yt', NewVar='yt_lag7')$yt_lag7
    yt_lag28 <- DataCombine::slide(train_min_max, slideBy=-28, Var='yt', NewVar='yt_lag28')$yt_lag28
    train_min_max$yt_lag1 <- yt_lag1
    train_min_max$yt_lag7 <- yt_lag7
    train_min_max$yt_lag28 <- yt_lag28
    x_train <- train_min_max[29:1913,-grep('yt', names(train_min_max))[1]]
    x_test <- train_min_max[1914:1941,-grep('yt', names(train_min_max))[1]]
    x_test$yt_lag7[8:28] = NA
    x_test$yt_lag1[2:28] = NA
    y_train <- train_min_max$yt[29:1913]
    train_nn <- cbind(y_train,x_train)
    head(train_nn)
    
    #train model
    epsilon <- 0.01
    repeat{
      layer <- c(5,5)
      af <- 'logistic'
      epsilon <- epsilon
      iter <- 10e5
      catch <- 0
      tryCatch(
        model_dlnn <- neuralnet(y_train~., data=train_nn, act.fct = af, threshold = epsilon,
                                stepmax = iter, linear.output = T, hidden = layer),
        warning = function(w) { catch <<- catch + 1 }
      )
      if(catch==0){
        break
      }
      epsilon <- epsilon*10
    }
    
    #forecast
    forecast_dlnn <- matrix(ncol=1,nrow=28)
    for(k in 1:28){
      if(is.na(x_test$yt_lag1[k])){x_test$yt_lag1[k]=forecast_dlnn[k-1]}
      if(is.na(x_test$yt_lag7[k])){x_test$yt_lag7[k]=forecast_dlnn[k-7]}
      forecast_dlnn[k] <- compute(model_dlnn, x_test[k,])$net.result
    }
    forecast_dlnn <- (forecast_dlnn*max(train$yt))
    sample_sub[,i] <- forecast_dlnn
    if (i == N) message("Done!")
  }
  return(sample_sub)
}
run_dlnn()
