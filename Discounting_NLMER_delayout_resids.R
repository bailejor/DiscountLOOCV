# Load required packages
library(nlme)
library(tidyverse)
library(nls.multstart)
library(lattice)


set.seed(1234)


# Read the delay-discounting data
df1 <- read.csv("mazur_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)
df2 <- read.csv("Rachlin_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)
df3 <- read.csv("MG_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)
df4 <- read.csv("exponential_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)
df5 <- read.csv("BD_sim_johnsonbickel.csv", check.names = FALSE, header = TRUE)

sample_df1 <- df1[sample(nrow(df1), 200), ]
sample_df2 <- df2[sample(nrow(df2), 200), ]
sample_df3 <- df3[sample(nrow(df3), 200), ]
sample_df4 <- df4[sample(nrow(df4), 200), ]
sample_df5 <- df5[sample(nrow(df5), 200), ]

# Combine the sampled data frames into one dataframe
df <- rbind(sample_df1, sample_df2, sample_df3, sample_df4, sample_df5)

# Check the dimensions of the combined dataframe
print(dim(df))


#df <- df[, c("1", "14", "30", "183", "365", "9125")]

#In reviewing the higher density set, the values are the same early in the delays, but at 9125 they depart drastically

               
#ADD PARTICIPANT ID NUMBER
df <- tibble::rowid_to_column(df, "ID")


#PIVOT FROM WIDE TO LONG FORM
df_long <- df %>%
  pivot_longer(cols = -ID, names_to = "Delay", values_to = "Value")
df_long$Delay = as.numeric(as.character(df_long$Delay))


#USE NLS_MULT START TO EXTRACT GUESSES AT STARTING PARAMETERS THEN NLME BELOW TO GET FIXED EFFECTS AND RANDOM EFFECTS

test_set <- head(df_long, 8000)


#########MAZUR#########################################################
mae_scores_mazur <- list()
score_df_mazur <- data.frame(MAE = numeric(), ID = numeric())

  
delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)
#delay_list <- list(1, 14, 30, 183, 365, 9125)


  for (i in delay_list){
    test_row <- subset(test_set, Delay == i)
    train_row <- subset(test_set, Delay != i)


    start_vals <- nls_multstart(Value ~ 1/(1+k*Delay), data = train_row,
                                     iter = 500,
                                     start_lower = c(k = -4),
                                     start_upper = c(k = 4))
    
    tryCatch({
    mazur_fit <- nlme(Value ~ 1/(1+k*Delay), data = train_row,
                      random = k ~ 1,
                      fixed = k ~ 1,
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
                                     minScale = .0000001,
                                     opt = "optim"))
    
    
    
    jor2 <- predict(mazur_fit, test_row, level = 1)
    print(jor2)
    
    score <- c(jor2 - test_row$Value)
    
    
    mae_scores_mazur[[length(mae_scores_mazur) +1]] <- score
    }, 
    
    error = function(e) {
      # Code to handle the error
      print("An error occurred:")
      print(e)
    })

  }





  #overall_mae_mazur <- mean(unlist(mae_scores))

####RACHLIN#########################################################
mae_scores_rach <- list()
score_df_rach <- data.frame(MAE = numeric(), ID = numeric())
failed_fits_rach <- 0

delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)
#delay_list <- list(1, 14, 30, 183, 365, 9125)

for (i in delay_list){
  test_row <- subset(test_set, Delay == i)
  train_row <- subset(test_set, Delay != i)
    
  tryCatch({
    start_vals <- nls_multstart(Value ~ 1/(1+10^k*Delay^s), data = train_row,
                                     iter = 500,
                                     start_lower = c(k = -4, s = .1),
                                     start_upper = c(k = 4, s = 4))
    

    rach_fit <- nlme(Value ~ 1/(1+10^k*Delay^s), data = train_row,
                      random = list((k + s ~ 1)),
                      fixed = list(k ~ 1, s ~ 1),
                      start = list(fixed = coef(start_vals)),
                      groups = ~ID,
                      method = "ML",
                      verbose = 0,
                      control = list(msMaxIter = 5000,
                                     niterEM = 5000,
                                     maxIter = 50000,
                                     pnlsTol = .0001,
                                     tolerance = .4,
                                     apVar = T,
                                     minScale = .0000001,
                                     opt = "optim"))
    
    
    
    jor2 <- predict(rach_fit, test_row, level = 1)
    print(jor2)
    
    score <- (jor2 - test_row$Value)

    mae_scores_rach[[length(mae_scores_rach) +1]] <- score
  },
  error = function(e) {
    # Code to handle the error
    print("An error occurred:")
    print(e)
    failed_fits_rach <<- failed_fits_rach + 1 
  })
  
}





####RACHLIN#########################################################

####MYERSON GREEN#########################################################
mae_scores_mg <- list()
score_df_mg <- data.frame(MAE = numeric(), ID = numeric())
failed_fits_mg <- 0


delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)
#delay_list <- list(1, 14, 30, 183, 365, 9125)

for (i in delay_list){
  test_row <- subset(test_set, Delay == i)
  train_row <- subset(test_set, Delay != i)
    
    start_vals <- nls_multstart(Value ~ 1/(1 + k*Delay)^s, data = train_row,
                                iter = 500,
                                start_lower = c(k = -4, s = .1),
                                start_upper = c(k = 4, s = 10))
    tryCatch({
    mg_fit <- nlme(Value ~ 1/(1+k*Delay)^s, data = train_row,
                      random = list((k + s ~ 1)),
                      fixed = list(k ~ 1, s ~ 1),
                      start = list(fixed = coef(start_vals)),
                      groups = ~ID,
                      method = "ML",
                      verbose = 0,
                      control = list(msMaxIter = 5000,
                                     niterEM = 5000,
                                     maxIter = 50000,
                                     pnlsTol = .0001,
                                     tolerance = .4,
                                     apVar = T,
                                     minScale = .0000001,
                                     opt = "optim"))
    
    jor2 <- predict(mg_fit, test_row, level = 1)
    print(jor2)
    
    score <- (jor2 - test_row$Value)
    

    mae_scores_mg[[length(mae_scores_mg) +1]] <- score
    },
    error = function(e) {
      # Code to handle the error
      print("An error occurred:")
      print(e)
      failed_fits_mg <<- failed_fits_mg + 1 
      mae_scores_mg[[length(mae_scores_mg) +1]] <<- 0
    })

}

  #jor14 <- na.omit(mae_scores)
  #mae_scores <- mae_scores[!is.na(mae_scores)]
  #overall_mae_mg <- "FAIL"





####MYERSON GREEN#########################################################

####SAMUELSON#########################################################
mae_scores_expo <- list()
score_df_expo <- data.frame(MAE = numeric(), ID = numeric())
failed_fits_expo <- 0

delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)
#delay_list <- list(1, 14, 30, 183, 365, 9125)

for (i in delay_list){
  test_row <- subset(test_set, Delay == i)
  train_row <- subset(test_set, Delay != i)
    
    
  tryCatch({
    start_vals <- nls_multstart(Value ~ exp(-k* Delay), data = train_row,
                                iter = 500,
                                start_lower = c(k = -4),
                                start_upper = c(k = 4))
    
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
    
 
    jor2 <- predict(expo_fit, test_row, level = 1)
    print(jor2)
    
    score <- (jor2 - test_row$Value)
    
      mae_scores_expo[[length(mae_scores_expo) +1]] <- score
      
  },
  error = function(e) {
    # Code to handle the error
    print("An error occurred:")
    print(e)
    failed_fits_expo <<- failed_fits_expo + 1
  })
      
    }
    

    
####SAMUELSON#########################################################

####BD#########################################################
mae_scores_bd <- list()
score_df_bd <- data.frame(MAE = numeric(), ID = numeric())
failed_fits_bd <- 0

delay_list <- list(1, 7, 14, 30, 183, 365, 1825, 9125)
#delay_list <- list(1, 14, 30, 183, 365, 9125)

for (i in delay_list){
  test_row <- subset(test_set, Delay == i)
  train_row <- subset(test_set, Delay != i)
  
    
    
    start_vals <- nls_multstart(Value ~ beta*exp(-delta*Delay), data = train_row,
                                iter = 500,
                                start_lower = c(beta = 0, delta = 0),
                                start_upper = c(beta = 1, delta = 1))
    
    tryCatch({
      BD_fit <- nlme(Value ~ beta*exp(-delta*Delay), data = train_row,
                        random = list(pdSymm(beta + delta ~ 1)),
                        fixed = list(beta ~ 1, delta ~ 1),
                        start = list(fixed = coef(start_vals)),
                        groups = ~ID,
                        method = "ML",
                        verbose = 0,
                        control = list(msMaxIter = 5000,
                                       niterEM = 5000,
                                       maxIter = 5000,
                                       pnlsTol = .0001,
                                       tolerance = .1,
                                       apVar = T,
                                       minScale = .0000001,
                                       opt = "optim"))
      
      
      jor2 <- predict(BD_fit, test_row, level = 1)
      print(jor2)
      
      score <- (jor2 - test_row$Value)


      mae_scores_bd[[length(mae_scores_bd) +1]] <- score
      
    },
    error = function(e) {
      # Code to handle the error
      print("An error occurred:")
      print(e)
      failed_fits_bd <<- failed_fits_bd + 1
    })
      
}


####BD#########################################################
#par(mfrow=c(2, 2))


par(mfrow=c(3,2), mar=c(1, 2, 1, 1), oma=c(2,2,0,0))
boxplot(mae_scores_mazur, delay_list, xaxt='n', main="Mazur", ylim = c(-1, 1))
#axis(side=1, at=1:10, labels=1:10)
boxplot(mae_scores_rach, delay_list, xaxt='n', yaxt = 'n', main="Rachlin", ylim = c(-1, 1))
boxplot(mae_scores_mg, delay_list, xaxt='n', main="MG", ylim = c(-1, 1))
boxplot(mae_scores_expo, delay_list, yaxt = 'n', xaxt = 'n', main="Samuelson", ylim = c(-1, 1))
mtext("Holdout Delay",side=1,line=3,outer=FALSE,cex=1.0, las = 1)
axis(side=1, at=1:8, labels=c(1, 7, 14, 30, 183, 365, 1825, 9125), las=2)
boxplot(mae_scores_bd, delay_list, main="Laibson", xaxt = 'n', ylim = c(-1, 1))
axis(side=1, at=1:8, labels=c(1, 7, 14, 30, 183, 365, 1825, 9125), las=2)


mtext("Residuals",side=2,line=0,outer=TRUE,cex=1.0,las=0)


#############COMBINE ALL SCORE DATAFRAMES#################################
##RUN IF FIRST ROW
score_list <- data.frame(overall_mae_mazur, overall_mae_rach, overall_mae_mg, overall_mae_expo, overall_mae_bd)
score_list_fail <- data.frame(failed_fits_mazur, failed_fits_rach, failed_fits_mg, failed_fits_expo, failed_fits_bd)
#colnames(score_list) <- c("Mazur_AIC", "Mazur_BIC", "Rach_AIC", "Rach_BIC", "MG_AIC", "MG_BIC", "Expo_AIC", "Expo_BIC", "BD_AIC", "BD_BIC")
##### ADD SUBSEQUENT ROWS####
new_row <- data.frame(overall_mae_mazur, overall_mae_rach, overall_mae_mg, overall_mae_expo, overall_mae_bd)
score_list <- rbind(score_list, new_row)

new_row_fail <- data.frame(failed_fits_mazur, failed_fits_rach, failed_fits_mg, failed_fits_expo, failed_fits_bd)
score_list_fail <- rbind(score_list_fail, new_row_fail)



write.csv(score_list, file = "LOOCVResults_6.csv", row.names = FALSE)
write.csv(score_list_fail, file = "LOOCVFailedFits_6.csv", row.names = FALSE)




