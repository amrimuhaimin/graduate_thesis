#load library
#general visualisation
library(ggplot2)
library(scales)
library(patchwork)
library(RColorBrewer)
library(corrplot)

#general data manipulation
library(dplyr)
library(readr)
library(vroom)
library(skimr)
library(tibble)
library(tidyr)
library(purrr)
library(stringr)
library(forcats)
library(fuzzyjoin)
install.packages('fastDummies');library(fastDummies)

# specific visualisation
library('alluvial') # visualisation
library('ggrepel') # visualisation
library('ggforce') # visualisation
library('ggridges') # visualisation
library('gganimate') # animations
library('GGally') # visualisation
library('ggthemes') # visualisation
library('wesanderson') # visualisation
library('kableExtra') # display
library(plotly)
library(hrbrthemes)

#modeling and forecast
library(tsintermittent)
library(Metrics)
library(kernlab)
library(caret)
library(e1071)
library(tscount)
library(neuralnet)

# helper function
get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

#load dataset
path = "E:/NCTU/Thesis Ref/Data/M5/"
train <- vroom(stringr::str_c(path,'sales_train_validation.csv'), delim=",", col_types = cols())
evaluation <- vroom(stringr::str_c(path, 'sales_train_evaluation.csv'), delim=',', col_types= cols())
prices <- vroom(stringr::str_c(path, 'sell_prices.csv'), delim=',', col_types = cols())
calendar <- readr::read_csv(stringr::str_c(path, 'calendar.csv'), col_types = cols())
sample_submit <- vroom(stringr::str_c(path, 'sample_submission.csv'), delim = ',', col_types = cols())

#get some random dataset 
df <- t(train[4,7:ncol(train)]) #train data
df <- data.frame(yt=df) #the random dataset
df <- cbind(df,calendar[1:1913,]) #df training
df_test <- t(evaluation[4,7:ncol(evaluation)]) #test data
df_test <- data.frame(yt=df_test)
df_test <- cbind(df_test,calendar[1:1941,])

#visualize the data
ggplot(data=df, aes(x=date,y=yt)) +
  geom_bar(stat="identity",color="#69b3a2") +
  ylab("Count Demand Sales") +
  theme_minimal()
barplot(prop.table(table(ifelse(df$yt > 0,"non-zero demand","zero demand"))), ylim=c(0,1), col="light blue")

demand_decomp <- data.frame(demand=crost.decomp(df$yt)$demand) #demand
interval_decomp <- data.frame(interval=crost.decomp(df$yt)$interval) #interval
ggplot(data=demand_decomp, aes(x=seq(1:nrow(demand_decomp)), y=demand)) +  #demand plot
  geom_bar(stat="identity",color="#69b3a2") +
  xlab("Index Demand") +
  theme_minimal()
ggplot(data=interval_decomp, aes(x=seq(1:nrow(interval_decomp)), y=interval)) + #interval plot
  geom_line(color="#69b3a2") +
  xlab("Index Interval") + 
  theme_minimal()
day_aggregate <- data.frame(day=c(aggregate(df$yt, by=list(df$weekday), sum)[1]),
                            value=c(aggregate(df$yt, by=list(df$weekday), sum)[2])) #day aggregate
colnames(day_aggregate) <- c("day", "value")
ggplot(day_aggregate, aes(x=factor(day, level=c("Monday","Tuesday","Wednesday","Thursday", #weekly plot
                                                "Friday","Saturday","Sunday")), y=value, group=1)) +
  geom_line(color="#69b3a2", size=2) + labs(x="", y="Sales Demand", title="Weekly Seasonality") +
  theme_minimal()
month_aggregate <- aggregate(yt~lubridate::month(date), data=df, FUN=sum)
colnames(month_aggregate) <- c("month", "value")
ggplot(month_aggregate, aes(x=factor(month, level=c(month.name)), y=value, group=1)) +#monthly seasonality
  geom_line(color="#69b3a2", size=2) +
  labs(x="", y="Sales Demand", title="Monthly Seasonality") + theme_minimal()
year_aggregate <- aggregate(yt~lubridate::year(date), data=df, FUN=sum)
colnames(year_aggregate) <- c('year', 'yt')
ggplot(year_aggregate, aes(x=year, y=yt)) + geom_line(color='#69b3a2', size=2) + #year plot trend
  labs(x="", y="Sales Demand", title="Trend Year") + theme_minimal()
yearmon_aggregate <- aggregate(yt~zoo::as.yearmon(date), data=df, FUN=sum)
yearmon_aggregate$idx <- seq(1:64)
colnames(yearmon_aggregate) <- c('yearmon', 'value', 'idx')
ggplot(yearmon_aggregate, aes(x=yearmon, y=value)) + geom_line(color="#69b3a2", size=2) +
  labs(x="", y="Demand Sales", title="Trend Year-Month") + theme_minimal()

#preprocessing the data
dummy_event1 <- ifelse(is.na(df$event_name_1)==TRUE,0,1)
dummy_event2 <- ifelse(is.na(df$event_name_2)==TRUE,0,1)
dummy_weekend <- ifelse(df$weekday=="Saturday" | df$weekday=="Sunday",1,0)
dummy_weekend <- fastDummies::dummy_cols(df, "weekday")
dummy_weekend <- dummy_weekend[,16:22]
dummy_snap <- df$snap_CA
dummy_month <- fastDummies::dummy_cols(df, "month")
dummy_month <- dummy_month[,16:27]
train <- data.frame(yt=df$yt, dummy_event1, dummy_event2, dummy_weekend, dummy_snap, dummy_month)
head(train)
testing <- df_test
testing <- data.frame(yt=testing$yt, dummy_snap=testing$snap_CA)
testing$dummy_event1 <- ifelse(is.na(df_test$event_name_1)==TRUE,0,1)
testing$dummy_event2 <- ifelse(is.na(df_test$event_name_2)==TRUE,0,1)
testing$dummy_weekend <- ifelse(df_test$weekday=="Saturday" | df_test$weekday=="Sunday", 0, 1)
dummy_weekend_testing <- fastDummies::dummy_cols(df_test, 'weekday')
dummy_weekend_testing <- dummy_weekend_testing[,16:22]
dummy_month_testing <- fastDummies::dummy_cols(df_test, 'month')
dummy_month_testing <- dummy_month_testing[,16:27]
testing <- cbind(testing, dummy_month_testing, dummy_weekend_testing)
testing <- testing[,-5]
testing <- testing[1914:1941,]
head(testing)
write.csv(train,'E:/NCTU/Thesis Ref/Data/train1.csv', row.names=F)
write.csv(testing,'E:/NCTU/Thesis Ref/Data/test1.csv', row.names=F)
train <- read.csv('E:/NCTU/Thesis Ref/Data/train1.csv', sep=',', header=T)
test <- read.csv('E:/NCTU/Thesis Ref/Data/test1.csv', sep=',', header=T)
df <- train
df_test <- test
#ad-hoc model croston
forecast_cr <- crost(df$yt, h = 28, type='croston') #optimize croston
forecast_sba <- crost(df$yt, h = 28, type='sba') #optimize sba
forecast_sbj <- crost(df$yt, h= 28, type='sbj') #optimize sbj
smape(train$yt[38:1913], forecast_cr$frc.in[38:1913]) #insample
smape(train$yt[38:1913], forecast_sba$frc.in[38:1913])
smape(train$yt[38:1913], forecast_sbj$frc.in[38:1913])
smape(df_test$yt, forecast_cr$frc.out) #outsample
smape(df_test$yt, forecast_sba$frc.out) #outsample
smape(df_test$yt, forecast_sbj$frc.out) #outsample

#model based
#time series regression
model_tsr <- lm(yt~weekday_Monday+weekday_Tuesday+weekday_Wednesday+weekday_Thursday+
                weekday_Friday+weekday_Saturday+weekday_Sunday-1, train)
summary(model_tsr)
ts.plot(model_tsr$residuals)
forecast_tsr <- predict(model_tsr, test[,-1])
forecast_tsr
smape(train$yt, model_tsr$fitted.values)
smape(test$yt, forecast_tsr) #outsample

#ARIMA
acf(df$yt)
acf(diff(df$yt, differences = 1))
pacf(df$yt)
model_arima <-Arima(df$yt, order = c(0,1,1), include.mean = F, seasonal = list(order=c(0,1,1), period=7),
                    method = c('ML'))
lmtest::coeftest(model_arima)
acf(model_arima$residuals)
pacf(model_arima$residuals)
forecast_arima <-as.data.frame(forecast(model_arima, h=28))
smape(train$yt[41:1913], model_arima$fitted[41:1913])
smape(test$yt, forecast_arima[,1]) #outsample
ts.plot(model_tsr$fitted.values)
min(model_arima$fitted)

#ARIMAX
acf(diff(model_tsr$residuals, period=1))
model_arimax <- arima(model_tsr$residuals, order=c(0,1,1), method="ML", transform.pars = F)
lmtest::coeftest(model_arimax)
forecast_arimax <- forecast(model_arimax, h=28)
inforecast_arimax2 <- model_tsr$fitted.values + as.vector(forecast_arimax$fitted)
forecast_arimax2 <- model_tsr$coefficients[1]*test$weekday_Monday + model_tsr$coefficients[2]*test$weekday_Tuesday +
  model_tsr$coefficients[3]*test$weekday_Wednesday + model_tsr$coefficients[4]*test$weekday_Thursday +
  model_tsr$coefficients[5]*test$weekday_Friday + model_tsr$coefficients[6]*test$weekday_Saturday +
  model_tsr$coefficients[7]*test$weekday_Sunday + as.vector(forecast_arimax$mean)
smape(train$yt, inforecast_arimax2)
smape(test$yt, forecast_arimax2) #outsample
ts.plot(inforecast_arimax2)

#ksvm
train_test <- rbind(train[,order(names(train))], test[,order(names(test))])
ylag <- data.frame(yt_lag1 <-  DataCombine::slide(train_test, slideBy=-1, Var='yt', NewVar='yt1')$yt1,
                  yt_lag2 <- DataCombine::slide(train_test, slideBy=-2, Var='yt', NewVar='yt2')$yt2,
                  yt_lag3 <- DataCombine::slide(train_test, slideBy=-3, Var='yt', NewVar='yt3')$yt3,
                  yt_lag4 <- DataCombine::slide(train_test, slideBy=-4, Var='yt', NewVar='yt4')$yt4,
                  yt_lag5 <- DataCombine::slide(train_test, slideBy=-5, Var='yt', NewVar='yt5')$yt5,
                  yt_lag6 <- DataCombine::slide(train_test, slideBy=-6, Var='yt', NewVar='yt6')$yt6,
                  yt_lag7 <- DataCombine::slide(train_test, slideBy=-7, Var='yt', NewVar='yt7')$yt7)
colnames(ylag) <- c('ytlag1','ytlag2','ytlag3','ytlag4','ytlag5','ytlag6','ytlag7')
train_test <- data.frame(train_test, ylag)
train_svm <- train_test[8:1913,]
test_svm <- train_test[1914:1941,]

#cross val for find best parameter
ts_cv_svm <- function(){
  index_cv <- sort(seq(4,8), decreasing=T)
  sigma_par <- c(4,8,16)
  eps_par <- c(0.01,0.1,1)
  cv_smape <- matrix(ncol=3,nrow=45)
  m <- 1
  for(j in sigma_par){
    for(k in eps_par){
      for(i in index_cv){
        train_cv <- train_svm[1:(nrow(train_svm)-i),]
        test_cv <- train_svm[-(1:(nrow(train_svm)-i)),]
        tuning_svm <- ksvm(x=cbind(train_cv$ytlag1,train_cv$ytlag2,train_cv$ytlag3,train_cv$ytlag4,
                                   train_cv$ytlag5,train_cv$ytlag6,train_cv$ytlag7,
                                   train_cv$weekday_Monday,train_cv$weekday_Friday,train_cv$weekday_Saturday,
                                   train_cv$weekday_Sunday,train_cv$weekday_Thursday,train_cv$weekday_Tuesday,
                                   train_cv$weekday_Wednesday), y=train_cv$yt,
                           epsilon=eps_par, sigma=sigma_par, cross=0, kernel='rbfdot', tol=0.001, type='eps-svr')
        forecast_cv <- predict(tuning_svm, cbind(test_cv$ytlag1,test_cv$ytlag2,test_cv$ytlag3,test_cv$ytlag4,
                                                 test_cv$ytlag5,test_cv$ytlag6,test_cv$ytlag7,
                                                 test_cv$weekday_Monday,test_cv$weekday_Friday,test_cv$weekday_Saturday,
                                                 test_cv$weekday_Sunday,test_cv$weekday_Thursday,test_cv$weekday_Tuesday,
                                                 test_cv$weekday_Wednesday))
        cv_smape[m,1] <- smape(test_cv$yt[1:4], forecast_cv[1:4])
        cv_smape[m,2] <- k
        cv_smape[m,3] <- j
        m <- m+1
      }
    }
  }
  return(cv_smape)
}
tuning.svm <- ts_cv_svm()
tuning.svm <- as.data.frame(tuning.svm)
tuning.svm[c('V2','V3')] <- lapply(tuning.svm[c('V2','V3')], as.factor)
tuning.svm %>%
  group_by(V2,V3) %>%
  summarise_at(vars("V1"), mean)
model_svm <- ksvm(x=cbind(train_svm$ytlag1,train_svm$ytlag2,train_svm$ytlag3,train_svm$ytlag4,
                          train_svm$ytlag5,train_svm$ytlag6,train_svm$ytlag7,
                          train_svm$weekday_Monday,train_svm$weekday_Friday,train_svm$weekday_Saturday,
                          train_svm$weekday_Sunday,train_svm$weekday_Thursday,train_svm$weekday_Tuesday,
                          train_svm$weekday_Wednesday), y=train_svm$yt,
                  epsilon=0.01, sigma=16, cross=0, kernel='rbfdot', tol=0.001, type='eps-svr')

#predict
test_p <- test_svm
test_p[8:28,names(ylag)] <- NA
test_p$ytlag1[2:28] <- NA
test_p$ytlag2[3:28] <- NA
test_p$ytlag3[4:28] <- NA
test_p$ytlag4[5:28] <- NA
test_p$ytlag5[6:28] <- NA
test_p$ytlag6[7:28] <- NA
test_p$yhat <- numeric(28)
for(i in 1:28){
  if(is.na(test_p[i,24]))
  {test_p[i,24] <- test_p$yhat[i-1]}
  if(is.na(test_p[i,25]))
  {test_p[i,25] <- test_p$yhat[i-2]}
  if(is.na(test_p[i,26]))
  {test_p[i,26] <- test_p$yhat[i-3]}
  if(is.na(test_p[i,27]))
  {test_p[i,27] <- test_p$yhat[i-4]}
  if(is.na(test_p[i,28]))
  {test_p[i,28] <- test_p$yhat[i-5]}
  if(is.na(test_p[i,29]))
  {test_p[i,29] <- test_p$yhat[i-6]}
  if(is.na(test_p[i,30]))
  {test_p[i,30] <- test_p$yhat[i-7]}
  test_p$yhat[i] <- predict(model_svm, cbind(test_p$ytlag1[i],test_p$ytlag2[i],test_p$ytlag3[i],test_p$ytlag4[i],
                                             test_p$ytlag5[i],test_p$ytlag6[i],test_p$ytlag7[i],
                                             test_p$weekday_Monday[i],test_p$weekday_Friday[i],test_p$weekday_Saturday[i],
                                             test_p$weekday_Sunday[i],test_p$weekday_Thursday[i],test_p$weekday_Tuesday[i],
                                             test_p$weekday_Wednesday[i]))
  
}
forecast_ksvm <- test_p$yhat
smape(train_svm$yt, model_svm@fitted)
smape(test$yt, test_p$yhat) #outsample
ts.plot(test_p$yhat)

#hybrid ksvm
train_res <- model_tsr$residuals
train_testx <- c(train_res, test$yt)
train_testx <- data.frame(yt=train_testx)
ylag<- data.frame(yt_lag1 <-  DataCombine::slide(train_testx, slideBy=-1, Var='yt', NewVar='yt1')$yt1,
                  yt_lag2 <- DataCombine::slide(train_testx, slideBy=-2, Var='yt', NewVar='yt2')$yt2,
                  yt_lag3 <- DataCombine::slide(train_testx, slideBy=-3, Var='yt', NewVar='yt3')$yt3)
colnames(ylag) <- c('ytlag1','ytlag2','ytlag3')
train_testx <- data.frame(train_testx, ylag)
train_svmx <- train_testx[4:1913,]
test_svmx <- train_testx[1914:1941,]
#tuning svmx
ts_cv_svmx <- function(){
  index_cv <- sort(seq(4,8), decreasing=T)
  sigma_par <- c(4,8,16)
  eps_par <- c(0.01,0.1,1)
  cv_smape <- matrix(ncol=3,nrow=45)
  m <- 1
  for(j in sigma_par){
    for(k in eps_par){
      for(i in index_cv){
        train_cv <- train_svmx[1:(nrow(train_svm)-i),]
        test_cv <- train_svmx[-(1:(nrow(train_svm)-i)),]
        tuning_svm <- ksvm(x=cbind(train_cv$ytlag1,train_cv$ytlag2,train_cv$ytlag3), y=train_cv$yt,
                           epsilon=eps_par, sigma=sigma_par, cross=0, kernel='rbfdot', tol=0.001, type='eps-svr')
        forecast_cv <- predict(tuning_svm, cbind(test_cv$ytlag1,test_cv$ytlag2,test_cv$ytlag3))
        cv_smape[m,1] <- smape(test_cv$yt[1:4], forecast_cv[1:4])
        cv_smape[m,2] <- k
        cv_smape[m,3] <- j
        m <- m+1
      }
    }
  }
  return(cv_smape)
}
tuning.svmx <- ts_cv_svmx()
tuning.svmx <- as.data.frame(tuning.svmx)
tuning.svmx[c('V2','V3')] <- lapply(tuning.svmx[c('V2','V3')], as.factor)
tuning.svmx %>%
  group_by(V2,V3) %>%
  summarise_at(vars("V1"), mean)
#predict ksvmx
model_svmx <- ksvm(x=cbind(train_svmx$ytlag1,train_svmx$ytlag2,train_svmx$ytlag3), y=train_svmx$yt,
                   epsilon=0.1, sigma=4, kernel='rbfdot', cross=0, tol=0.001, type='eps-svr')
test_px <- test_svmx
test_px$ytlag1[2:28] <- NA
test_px$ytlag2[3:28] <- NA
test_px$ytlag3[4:28] <- NA
test_px$yhat <- numeric(28)
forecast_ksvmx <- model_tsr$coefficients[1]*test$weekday_Monday + model_tsr$coefficients[2]*test$weekday_Tuesday +
  model_tsr$coefficients[3]*test$weekday_Wednesday + model_tsr$coefficients[4]*test$weekday_Thursday +
  model_tsr$coefficients[5]*test$weekday_Friday + model_tsr$coefficients[6]*test$weekday_Saturday +
  model_tsr$coefficients[7]*test$weekday_Sunday + as.vector(test_px$yhat) #outsample
inforecast_ksvmx <- model_tsr$fitted.values[4:1913] + model_svmx@fitted
smape(train$yt[4:1913],inforecast_ksvmx) #outsample
smape(test$yt, forecast_ksvmx)


for(i in 1:28){
  if(is.na(test_px[i,2]))
  {test_px[i,2] <- test_px$yhat[i-1]}
  if(is.na(test_px[i,3]))
  {test_px[i,3] <- test_px$yhat[i-2]}
  if(is.na(test_px[i,4]))
  {test_px[i,4] <- test_px$yhat[i-3]}
  test_px$yhat[i] <- predict(model_svmx, cbind(test_px$ytlag1[i],test_px$ytlag2[i],test_px$ytlag3[i]))
}
test_px$yhat
test_px
forecast_

#deep learning with 2 hidden layer (DLNN)

 
#visualize after modeling
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_cr$frc.out)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="Croston Method") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_sba$frc.out)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="Croston-SBA Method") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_sbj$frc.out)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="Croston-SBJ Method") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_tsr)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="TSR") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_arima$`Point Forecast`)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="ARIMA") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_arimax2)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="ARIMAX") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=forecast_svm)) + geom_line(color="blue", size=2) +
  labs(x="", y="Demand Forecast", title="SVM") + theme_minimal()
ggplot(df_test[1914:1941,],aes(x=date,y=yt)) + geom_line(color="blue", size=2) +
  labs(x="", y="Validation Forecast", title="Real") + theme_minimal()
head(df)
tail(df)
