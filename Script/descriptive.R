#load library
library(ggplot2)

#load dataset
getwd()
dataset1 <- data.frame(yt=c(t(read.csv("df_experiment_test2.csv", header=T, sep=',')[1,7:1947]))); head(dataset1)
dataset2 <- data.frame(yt=c(t(read.csv("df_experiment_test2.csv", header=T, sep=',')[2,7:1947]))); head(dataset2)
dataset3 <- data.frame(yt=c(t(read.csv("df_experiment_test2.csv", header=T, sep=',')[3,7:1947]))); head(dataset3)

#plot the data
ggplot(dataset1, aes(x=seq(1:nrow(dataset1)), y=yt)) + geom_line(size=0.1) + ylab("day") + xlab("demand rate") +
  ggtitle("Dataset 1") + theme_classic()

ggplot(dataset2, aes(x=seq(1:nrow(dataset2)), y=yt)) + geom_line(size=0.1) + ylab("day") + xlab("demand rate") +
  ggtitle("Dataset 2") + theme_classic()

ggplot(dataset3, aes(x=seq(1:nrow(dataset3)), y=yt)) + geom_line(size=0.1) + ylab("day") + xlab("demand rate") +
  ggtitle("Dataset 3") + theme_classic()

#check the charasteristic from the data
tsintermittent::idclass(dataset1$yt)
tsintermittent::idclass(dataset2$yt)
tsintermittent::idclass(dataset3$yt)

#result gru
plot(test_data$yt[28:1941], col='blue', type='l', lwd=2, ylab="demand rate", xlab="time/day")
lines(f_lstm[[2]], col= 'red', lwd=2)
lines(c(rep(NA, 1885), f_lstm[[1]]), col= 'green', lwd=2)


#check demand level
sum(dataset1$yt)/sum(ifelse(dataset1$yt > 1, 1 , 0))
sum(dataset2$yt)/sum(ifelse(dataset2$yt > 1, 1 , 0))
sum(dataset3$yt)/sum(ifelse(dataset3$yt > 1, 1 , 0))
