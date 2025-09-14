library(loo)
library(tidyverse)

BM_without <- read_rds("~/Desktop/Jules/fit_BM_no_age-sex.RDS")
OU_without <- read_rds("~/Desktop/Jules/fit_OU_no_age-sex.RDS")

BM_with <- read_rds("~/Desktop/Jules/fit_BM_with_age-sex.RDS")
OU_with <- read_rds("~/Desktop/Jules/fit_OU_with_age-sex.RDS")

elpds <- loo_compare(
  list(
    "BM_without"=BM_without$loo(), 
    "OU_without"=OU_without$loo(),
    "BM_with"=BM_with$loo(), 
    "OU_with"=OU_with$loo()
    )
)

saveRDS(elpds, "~/Desktop/Jules/mod_comp.rds")

model_weights <- 
  loo_model_weights(
    list(
      "BM_without"=BM_without$loo(), 
      "OU_without"=OU_without$loo(),
      "BM_with"=BM_with$loo(), 
      "OU_with"=OU_with$loo()
    )
  )

saveRDS(model_weights, "~/Desktop/Jules/weights.rds")
