###simulaton study####
library(tsintermittent)

sim1 <- simID(n=1, obs=(365*5), idi=15, cv2=0.5, level=5);ts.plot(sim1)
sim2 <- simID(n=1, obs=(365*5), idi=15, cv2=0.5, level=2);ts.plot(sim2)
sim3 <- simID(n=1, obs=(365*5), idi=5, cv2=0.5, level=5);ts.plot(sim3)
sim4 <- simID(n=1, obs=(365*5), idi=5, cv2=0.5, level=2);ts.plot(sim4)

train_data_sim <- data.frame(yt=c(sim1$ts.1[1:(1825-28)]))
test_data_sim <- data.frame(yt=c(sim1$ts.1))

f_crost_sim_sbj <- crost(train_data_sim, h=28, type='sbj')
f_crost_sim_sba <- crost(train_data_sim, h=28, type='sba')
f_gru <- model.gru_nonrc(train_data=train_data_sim, test_data = test_data_sim, lag=28, batch_size=64,
                   val_split=NULL, shuffle=F, optimizer='rmsprop', gru_units=128, rec_dropout=0.2, dropout=0.2,
                   epochs=100, out_act='sigmoid')
f_lstm <- model.lstm_nonrc(train_data_sim, test_data_sim, lag=28, batch_size=32, optimizer = 'adam', lstm_units = 128,
                           epochs=100, val_split=NULL, shuffle=F, dropout = 0.1, rec_dropout = 0.1,
                           out_act = 'sigmoid', verbose=0)
f_lstm <- model.lstm(train_data_sim, test_data_sim, lag=28, batch_size=32, optimizer = 'adam', lstm_units = 128,
                     epochs=100, val_split=NULL, shuffle=F, dropout = 0.2, rec_dropout = 0.2,
                     out_act = 'sigmoid')

eval_model <- function(te_data, tr_data, f_data, h){
  forecast_data <-c(tr_data, f_data)
  return(RMSSE(te_data, forecast_data, h))
}
eval_model(test_data_sim$yt, train_data_sim$yt, f_crost_sim_sbj$frc.out, h=28)
eval_model(test_data_sim$yt, train_data_sim$yt, f_crost_sim_sba$frc.out, h=28)


ts.plot(test_data_sim$yt[1797:1825])
f_crost_sim_sba$frc.out
abs(sum(f_crost_sim_sba$frc.out)-sum(tail(test_data_sim$yt,28)))
abs(sum(f_crost_sim_sbj$frc.out)-sum(tail(test_data_sim$yt,28)))
abs(sum(f_gru)-sum(tail(test_data_sim$yt,28)))
