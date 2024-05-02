# Load required packages
library(nlme)
library(tidyverse)
library(nls.multstart)


set.seed(1234)


# Read the delay-discounting data
df <- read.csv("BD_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)

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
delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)



test_set <- head(df_long, 8000)


  
  
  start_vals <- coef(nls_multstart(Value ~ 1/(1+k*Delay), data = test_set,
                                   iter = 500,
                                   start_lower = c(k = -4),
                                   start_upper = c(k = 4)))
  
  
  
  
  mazur_fit <- nlme(Value ~ 1/(1+k*Delay), data = test_set,
                    random = k ~ 1,
                    fixed = k ~ 1,
                    start = start_vals,
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
  
  print(jor11)
  jor11 <- predict(mazur_fit, test_row, level = 0)
  plot.new()
  plot(delay_list, jor11)
  par(mfrow=c(1,1))
  
  aic_value_mazur <- AIC(mazur_fit)
  bic_value_mazur <- BIC(mazur_fit)
  
  print(aic_value_mazur)
  print(bic_value_mazur)

  
  
##############################RACHLIN#######################################
  start_vals <- nls_multstart(Value ~ 1/(1+10^k*Delay^s), data = test_set,
                              iter = 500,
                              start_lower = c(k = -4, s = .1),
                              start_upper = c(k = 4, s = 4))
  
  print(coef(start_vals))
  
  rach_fit <- nlme(Value ~ 1/(1+10^k*Delay^s), data = test_set,
                    random = list((k + s ~ 1)),
                    fixed = list(k ~ 1, s ~ 1),
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
  
  aic_value_rach <- AIC(rach_fit)
  bic_value_rach <- BIC(rach_fit)
  
  print(aic_value_rach)
  print(bic_value_rach)
  
  
############################MG###########################################
  
  start_vals <- nls_multstart(Value ~ 1/(1+k*Delay)^s, data = test_set,
                              iter = 500,
                              start_lower = c(k = -4, s = .1),
                              start_upper = c(k = 4, s = 4))
  
  print(start_vals)
  
  
  
  mg_fit <- nlme(Value ~ 1/(1+k*Delay)^s, data = test_set,
                    random = list((k + s ~ 1)),
                    fixed = list(k ~ 1, s ~ 1),
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
  
  
  
  #############################################################################
  jor22 <- predict(rach_fit, test_set, level = 1)
  jor22 <- unlist(jor22)
  
  jor33 <- predict(mazur_fit, test_set, level = 1)
  jor33 <- unlist(jor33)
  
  jor44 <- predict(bd_fit, test_set, level = 1)
  jor44 <- unlist(jor44)
  
  jor55 <- predict(expo_fit, test_set, level = 1)
  jor55 <- unlist(jor55)

  test_set$rach_preds <- jor22
  test_set$mazur_preds <- jor33
  test_set$BD_preds <- jor44
  test_set$expo_preds <- jor55
  

  
  # Get the list of unique participants
  participants <- unique(test_set$ID)
  
  par(mfrow=c(4,1))
  plot(1, 1, type = "n", xlim = range(test_set$Delay), ylim = range(test_set$rach_preds), xlab = "X", ylab = "Y")
  
  # Add lines to the plot for each participant
  for(participant in participants) {
    # Subset the data for the current participant
    subset_df <- test_set[test_set$ID == participant, ]
    
    # Plot points
    #points(subset_df$Delay, subset_df$rach_preds)
    
    # Connect points with lines
    lines(subset_df$Delay, subset_df$rach_preds, lwd = 0.1, col = "black")
  }
  
  plot(2, 1, type = "n", xlim = range(test_set$Delay), ylim = range(test_set$rach_preds), xlab = "X", ylab = "Y")
  for(participant in participants) {
    # Subset the data for the current participant
    subset_df <- test_set[test_set$ID == participant, ]
    
    # Plot points
    #points(subset_df$Delay, subset_df$rach_preds)
    
    # Connect points with lines
    lines(subset_df$Delay, subset_df$mazur_preds, lwd = 0.1, col = "blue", xlab = "X", ylab = "Y")
  }
  
  plot(2, 1, type = "n", xlim = range(test_set$Delay), ylim = range(test_set$rach_preds), xlab = "X", ylab = "Y")
  for(participant in participants) {
    # Subset the data for the current participant
    subset_df <- test_set[test_set$ID == participant, ]
    
    # Plot points
    #points(subset_df$Delay, subset_df$rach_preds)
    
    # Connect points with lines
    lines(subset_df$Delay, subset_df$BD_preds, lwd = 0.1, col = "red", xlab = "X", ylab = "Y")
  }
  
  plot(2, 1, type = "n", xlim = range(test_set$Delay), ylim = range(test_set$rach_preds), xlab = "X", ylab = "Y")
  for(participant in participants) {
    # Subset the data for the current participant
    subset_df <- test_set[test_set$ID == participant, ]
    
    # Plot points
    #points(subset_df$Delay, subset_df$rach_preds)
    
    # Connect points with lines
    lines(subset_df$Delay, subset_df$expo_preds, lwd = 0.1, col = "pink", xlab = "X", ylab = "Y")
  }
  
  