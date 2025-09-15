data <- read.table("predict.txt", header = TRUE, sep = "\t", stringsAsFactors = FALSE)
data$Group <- as.factor(data[[1]])
data$Value <- as.numeric(data[[2]])
t_result <- t.test(Value ~ Group, data = data, var.equal = TRUE)
p <- t_result$p.value
cat("p-value:", p)


