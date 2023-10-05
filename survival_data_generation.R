library(survival)
library(lightgbm)
library(data.table)


### simulation setting
n <- 1000; p <- 100
tt <- round(.8*n)
tt2 <- round(.9*n)
beta <- c(rep(1,10),rep(0, p - 10))
x <- matrix(rnorm(n*p), n, p)
real_time <- -(log(runif(n)))/(10*exp(drop(x %*% beta)))
cens_time <- rexp(n, rate = 1/10)
status <- as.numeric(real_time <= cens_time)
surv_time <- Surv(real_time, status)
surv_time_boost <-  2 * real_time * (status - .5)

### survival time simulation,
x_train <- x[seq_len(tt), ]
y_train <- surv_time[seq_len(tt)]
y_train_boost <- surv_time_boost[seq_len(tt)]
x_valid <- x[(tt + 1): tt2, ]
y_valid <- surv_time[(tt + 1): tt2]
y_valid_boost <- surv_time_boost[(tt + 1): tt2]
x_test <- x[(tt2 + 1): n, ]
y_test <- surv_time[(tt2 + 1): n]
y_test_boost <- surv_time_boost[(tt2 + 1): n]

### Convert data to data.table
d_train <- as.data.table(x_train)
d_valid <- as.data.table(x_valid)
d_test <- as.data.table(x_test)

ytr <- data.table(ytrain = y_train_boost)
yva <- data.table(yvalid = y_valid_boost)
yte <- data.table(ytest = y_test_boost)

### Save data to csv files
fwrite(d_train, 'xtr.csv')
fwrite(d_valid, 'xva.csv')
fwrite(d_test, 'xte.csv')
fwrite(ytr, 'ytr.csv')
fwrite(yva, 'yva.csv')
fwrite(yte, 'yte.csv')
