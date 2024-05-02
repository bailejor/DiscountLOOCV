# Load required packages
library(nlme)
library(tidyverse)
library(nls.multstart)


set.seed(1234)


# Read the delay-discounting data
df <- read.csv("mazur_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)

#Create subset of original 8 delays
#df <- df[, c("1", "14", "30", "183", "365", "9125")]


#In reviewing the higher density set, the values are the same early in the delays, but at 9125 they depart drastically


#ADD PARTICIPANT ID NUMBER
df <- tibble::rowid_to_column(df, "ID")


#PIVOT FROM WIDE TO LONG FORM
df_long <- df %>%
  pivot_longer(cols = -ID, names_to = "Delay", values_to = "Value")
df_long$Delay = as.numeric(as.character(df_long$Delay))

####EMPTY THE SCORING DF
score_df <- data.frame()




test_set <- head(df_long, 8000)


mae_scores_mazur <- list()  




k_list <- seq(6.110^-6, 6.01^4, length.out = 25)
k_dual <- seq(6.110^-6, 6.01^4, length.out = 10)
s_list <- seq(0.1, 10, length.out = 10)
 
#k_list <- seq(6.1*10^-6, 6.0*10^4, by = 0.5)
  
for (j in k_list){

  
  tryCatch({
  mazur_fit <- nlme(Value ~ 1/(1+k*Delay), data = test_set,
                    random = k ~ 1,
                    fixed = k ~ 1,
                    start = j,
                    groups = ~ID,
                    method = "ML",
                    verbose = 0,
                    control = list(msMaxIter = 5000,
                                   niterEM = 5000,
                                   maxIter = 5000,
                                   pnlsTol = .0001,
                                   tolerance = .4,
                                   apVar = T,
                                   minScale = .0000001,
                                   opt = "optim"))
  print(AIC(mazur_fit))
  mae_scores_mazur[[length(mae_scores_mazur) +1]] <- AIC(mazur_fit)
  },
  error = function(e) {
    # Code to handle the error
    print("An error occurred:")
    print(e)
  })
  }
  #aic_value_mazur <- AIC(mazur_fit)
  #bic_value_mazur <- BIC(mazur_fit)
  
  #print(aic_value_mazur)
  #print(bic_value_mazur)

  
  
##############################RACHLIN#######################################

mae_scores_rach <- list()  
for (k_ in k_dual){
  for (s_ in s_list){
  
    
    tryCatch({
  rach_fit <- nlme(Value ~ 1/(1+k*Delay^s), data = test_set,
                    random = list((k + s ~ 1)),
                    fixed = list(k ~ 1, s ~ 1),
                    start =  c(k = k_, s = s_),
                    groups = ~ID,
                    method = "ML",
                    verbose = 0,
                    control = list(msMaxIter = 5000,
                                   niterEM = 5000,
                                   maxIter = 5000,
                                   pnlsTol = .0001,
                                   tolerance = .4,
                                   apVar = T,
                                   minScale = .000001,
                                   opt = "optim"))
  
  aic_value_rach <- AIC(rach_fit)
  #bic_value_rach <- BIC(rach_fit)
  
  print(aic_value_rach)
  #print(bic_value_rach)
  mae_scores_rach[[length(mae_scores_rach) +1]] <- AIC(rach_fit)
  
  },
  error = function(e) {
    # Code to handle the error
    print("An error occurred:")
    print(e)
  })

  }}
  
############################MG###########################################
  
  start_vals <- nls_multstart(Value ~ 1/(1+k*Delay)^s, data = test_set,
                              iter = 500,
                              start_lower = c(k = -4, s = .1),
                              start_upper = c(k = 4, s = 4))
  
  print(start_vals)
  
  
  
  mg_fit <- nlme(Value ~ 1/(1+k*Delay)^s, data = test_set,
                    random = list((k + s ~ 1)),
                    fixed = list(k ~ 1, s ~ 1),
                    start = coef(start_vals),
                    groups = ~ID,
                    method = "ML",
                    verbose = 0,
                    control = list(msMaxIter = 5000,
                                   niterEM = 5000,
                                   maxIter = 5000,
                                   pnlsTol = .0001,
                                   tolerance = .4,
                                   apVar = T,
                                   minScale = .000001,
                                   opt = "optim"))
  
  aic_value_mg <-AIC(mg_fit)
  bic_value_mg <- BIC(mg_fit)
  
  #aic_value_mg <-"FAIL"
  #bic_value_mg <- "FAIL"
  
  print(aic_value)
  print(bic_value)
  
  
############################SAMUELSON###########################################
  
start_vals <- nls_multstart(Value ~ exp(-k* Delay), data = test_set,
                              iter = 500,
                              start_lower = c(k = -4),
                              start_upper = c(k = 4))
  
  print(start_vals)
  
  
  
  expo_fit <- nlme(Value ~ exp(-k* Delay), data = test_set,
                 random = k ~ 1,
                 fixed = k ~ 1,
                 start = list(fixed = coef(start_vals)),
                 groups = ~ID,
                 method = "ML",
                 verbose = 0,
                 control = list(msMaxIter = 5000,
                                niterEM = 5000,
                                maxIter = 5000,
                                pnlsTol = .0001,
                                tolerance = .4,
                                apVar = T,
                                minScale = .000001,
                                opt = "optim"))
  
  aic_value_expo <- AIC(expo_fit)
  bic_value_expo <- BIC(expo_fit)
  
  #aic_value_expo <- "FAIL"
  #bic_value_expo <- "FAIL"
  
  print(aic_value_expo)
  print(bic_value_expo)
  
  ############################BD###########################################
  
  start_vals <- nls_multstart(Value ~ beta*exp(-delta*Delay), data = train_row,
                              iter = c(30, 20),
                              start_lower = c(beta = 0, delta = 0),
                              start_upper = c(beta = 1, delta = 1))
  
  
  bd_fit <- nlme(Value ~ beta*exp(-delta*Delay), data = train_row,
                    random = list((beta + delta ~ 1)),
                    fixed = list(beta ~ 1, delta ~ 1),
                    start = list(fixed = coef(start_vals)),
                    groups = ~ID,
                    method = "ML",
                    verbose = 0,
                    control = list(msMaxIter = 5000,
                                   niterEM = 5000,
                                   maxIter = 5000,
                                   pnlsTol = .0001,
                                   tolerance = .4,
                                   apVar = T,
                                   minScale = .0000001,
                                   opt = "optim"))
  
  aic_value_bd <- AIC(bd_fit)
  bic_value_bd <- BIC(bd_fit)
  
  print(aic_value_bd)
  print(bic_value_bd)
  
  ##############################################################################
  ##RUN IF FIRST ROW
  score_list <- data.frame(aic_value_mazur, bic_value_mazur, aic_value_rach, bic_value_rach, aic_value_mg, bic_value_mg, aic_value_expo, bic_value_expo, aic_value_bd, bic_value_bd)
  #colnames(score_list) <- c("Mazur_AIC", "Mazur_BIC", "Rach_AIC", "Rach_BIC", "MG_AIC", "MG_BIC", "Expo_AIC", "Expo_BIC", "BD_AIC", "BD_BIC")
  ##### ADD SUBSEQUENT ROWS####
  new_row <- data.frame(aic_value_mazur, bic_value_mazur, aic_value_rach, bic_value_rach, aic_value_mg, bic_value_mg, aic_value_expo, bic_value_expo, aic_value_bd, bic_value_bd)
  score_list <- rbind(score_list, new_row)
  
  write.csv(score_list, file = "AIC&BICResults_6.csv", row.names = FALSE)
  
  
  