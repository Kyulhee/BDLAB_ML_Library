col_sub <- function(vec){
  #vec <- clinical_result[,1]
  for(i in 0:5){
    vec <- gsub(i, subtypes[i+1], vec)
  }
  return(vec)
}

result_path <- "C:/test/mucinous_training/TCGA_mucinous_DNN_space/results/"
save_result_path <- "C:/test/mucinous_training/TCGA_mucinous_DNN_space/result_sum/"

for(dir in c("old_set", "new_set")){
  setwd(paste0(result_path,dir))
  output <- list.files(pattern = "*.csv")
  model_names_1 <- NULL
  model_names_2 <- NULL
  clinical_result <- NULL
  acc_list <- NULL
  for(o in output){
    file_name <- unlist(strsplit(o, split = ".", fixed = TRUE))[1]
    df <- read.csv(o, header =TRUE)
    TP_box <- NULL
    FP_box <- NULL
    TN_box <- NULL
    FN_box <- NULL
    subtypes <- c("CESC", "COAD", "PAAD", "STAD", "UCEC", "UCS")
    
    if(ncol(df)==9){
      for(i in 0:(length(unique(df$y))-1)){
        #table(df$prediction, df$y)
        TP <- table(df$prediction, df$y)[i+1,i+1]
        FP <- sum(table(df$prediction, df$y)[i+1,-(i+1)])
        TN <- sum(table(df$prediction, df$y)[-(i+1),-(i+1)])
        FN <- sum(table(df$prediction, df$y)[-(i+1),i+1])
        tot <- sum(TP, FP, TN, FN)
        
        TP_box <- c(TP_box, TP)
        FP_box <- c(FP_box, FP)
        TN_box <- c(TN_box, TN)
        FN_box <- c(FN_box, FN)
      }
      df_result1 <- data.frame(Subtype=subtypes[1:length(unique(df$y))], TP=TP_box, FP=FP_box, TN=TN_box, FN=FN_box, row.names = NULL)
      df_result2 <- as.data.frame.matrix(table(df$prediction, df$y))
      df_result2 <- cbind(df_result2, apply(df_result2, 1, sum))
      df_result2 <- rbind(df_result2, apply(df_result2, 2, sum))
      names(df_result2) <- c(subtypes[1:length(unique(df$y))], "Total")
      subtype <- c(subtypes[1:length(unique(df$prediction))], "Total")
      df_result2 <- cbind(subtype, df_result2)
      acc <- nrow(df[df$prediction == df$y,]) / length(df$y)
      
      write.csv(df_result1, paste0(save_result_path,dir,"/contingency_",o), row.names = FALSE, quote =FALSE)
      write.csv(df_result2, paste0(save_result_path,dir,"/whole_",o), row.names = FALSE, quote =FALSE)
      
      acc_list <- c(acc_list, acc)
      model_names_1 <- c(model_names_1, file_name)
    }else{
      if(is.null(clinical_result)){
        clinical_result <- data.frame(df$prediction)
      }else{
        clinical_result <- cbind(clinical_result, df$prediction)
      }
      model_names_2 <- c(model_names_2, file_name)
    }
  }
  clinical_result <- as.data.frame(apply(clinical_result, 2, col_sub))
  names(clinical_result) <- model_names_2
  write.csv(clinical_result, paste0(save_result_path,dir,"/clinical_output_",dir,".csv"), row.names = FALSE, quote = FALSE)
  acc_table <- data.frame(model=model_names_1, acc=acc_list)
  write.csv(acc_table, paste0(save_result_path,dir,"/acc_table_",dir,".csv"), row.names = FALSE, quote = FALSE)
}

