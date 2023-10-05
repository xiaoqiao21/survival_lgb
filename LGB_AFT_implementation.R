library(survival)
library(lightgbm)
library(data.table)
source('lgb_loss_func.R')

d_train <- fread('xtr.csv')
d_valid <- fread('xva.csv')
d_test <- fread('xte.csv')
ytr <- fread('ytr.csv')
yva <- fread('yva.csv')
yte <- fread('yte.csv')
x_train <- as.matrix(d_train)
x_valid <- as.matrix(d_valid)
x_test <- as.matrix(d_test)

LDtrain <- lgb.Dataset(x_train, label = ytr[, ytrain])
LDvalid <- list(test = lgb.Dataset.create.valid(LDtrain, x_valid, label = yva[, yvalid]))
y_test <- yte[, ytest]

params <- list(eta = .01, lambda = .01, alpha = .01, subsample = .5, colsample_bytree = .5)

### Run LGB AFT log-normal model
model_lgb_aft_lognormal <- lgb.train(params, LDtrain, nround = 1000, obj = aft_lognormal_obj,
                                     eval = cidx_lgb_func, valids = LDvalid, verbose = -1,
                                     early_stopping_rounds = 20)

### Run LGB AFT Exponential model
model_lgb_aft_exponential <- lgb.train(params, LDtrain, nround = 1000, obj = aft_exponential_obj,
                                       eval = cidx_lgb_func, valids = LDvalid, verbose = -1,
                                       early_stopping_rounds = 20)

### Run LGB AFT Weibull model
model_lgb_aft_weibull <- lgb.train(params, LDtrain, nround = 1000, obj = aft_weibull_obj,
                                   eval = cidx_lgb_func, valids = LDvalid, verbose = -1,
                                   early_stopping_rounds = 20)


### Predict LGB AFT log-normal model
y_lgb_aft_lognormal_preds <- predict(model_lgb_aft_lognormal, x_test)

### Predict LGB AFT Exponential model
y_lgb_aft_exponential_preds <- predict(model_lgb_aft_exponential, x_test)

### Predict LGB AFT Weibull model
y_lgb_aft_weibull_preds <- predict(model_lgb_aft_weibull, x_test)


### Calculate C-index
cidx_result <- c('LGB_AFT_Lognormal' = concordance(y_test ~ y_lgb_aft_lognormal_preds)$con,
                 'LGB_AFT_Exponential' = concordance(y_test ~ y_lgb_aft_exponential_preds)$con,
                 'LGB_AFT_Weibull' = concordance(y_test ~ y_lgb_aft_weibull_preds)$con)

### obtain result
result <- data.table(model = names(cidx_result), 'C' = cidx_result)
result
