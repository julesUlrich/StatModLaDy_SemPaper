library(cmdstanr)
library(tidyverse)
library(phytools)

data <- read_csv("data/DoReCo_data_full.csv") |>
  select(-c(entropy_rate, speech_rate, count)) |>
  mutate(
    sex_s = if_else(sex == "m", 0.5, if_else(sex == "f", -0.5, NA_real_)),
    age_s = scale(age, center = TRUE, scale = TRUE)[, 1],
    lang_id = as.integer(as.factor(glottocode)),
    speaker_id = as.integer(as.factor(speaker)),
    conversation_id = as.integer(as.factor(conversation)),
    phoneme_id = as.integer(as.factor(phoneme))
  )

# Fill in NAs randomly
set.seed(42) # for reproducibility
data$sex_s[is.na(data$sex_s)] <- sample(c(-0.5, 0.5), sum(is.na(data$sex_s)), replace = TRUE)
data$age_s[is.na(data$age_s)] <- 0

# --- Make speaker-level vectors (length = N_speaker) ---
speaker_key <- data |>
  distinct(speaker_id, speaker, sex_s, age_s) |>
  arrange(speaker_id)

N <- nrow(data)
N_lang <- max(data$lang_id)
N_speaker <- max(data$speaker_id)
N_phoneme <- max(data$phoneme_id)
N_conv <- max(data$conversation_id)

sex_s_vec <- rep(NA_real_, N_speaker)
age_s_vec <- rep(NA_real_, N_speaker)
sex_s_vec[speaker_key$speaker_id] <- speaker_key$sex_s
age_s_vec[speaker_key$speaker_id] <- speaker_key$age_s


# --- Build phylo VCV in the SAME ORDER as lang_id ---
tree <- read_rds("data/phylo.rds")
tree$tip.label <- unique(data$glottocode)

# Create phylo distance matrix
VCV_BM <- vcv(tree, corr = T)

data_list <- list(
  N = N,
  N_lang = N_lang,
  N_speaker = N_speaker,
  N_phoneme = N_phoneme,
  N_conversation = N_conv,
  lang_id = data$lang_id,
  speaker_id = data$speaker_id,
  conversation_id = data$conversation_id,
  phoneme_id = data$phoneme_id,
  information_rate = data$information_rate,

  # SPEAKER-LEVEL covariates (length N_speaker!)
  sex_s = sex_s_vec,
  age_s = age_s_vec,
  VCV_BM = VCV_BM,
  prior_sim = 0
)

data_list$age_s

###############################
#### (3) Fit Phylogenetic model ##

## Compile stan code
PGLMM_BM <- cmdstan_model(here::here("models/BM_no_age-sex.stan"))

# Sample
fit <- PGLMM_BM$sample(
  data = data_list,
  chains = 4,
  parallel_chains = 4,
  init = 0.1,
  iter_warmup = 1000,
  iter_sampling = 7000,
  adapt_delta = 0.9,
  max_treedepth = 12
)

# Save fit
fit$save_object(file = "results/fit_BM_no_age-sex.RDS")

###############################
#### (4) Fit Phylogenetic model with Age & Sex ##

## Compile stan code
PGLMM_BMA <- cmdstan_model(here::here("models/BM_with_age-sex.stan"))

# Sample
fit2 <- PGLMM_BMA$sample(
  data = data_list,
  chains = 4,
  parallel_chains = 4,
  init = 0.1,
  iter_warmup = 1000,
  iter_sampling = 7000,
  adapt_delta = 0.9,
  max_treedepth = 12
)

# Save fit
fit2$save_object(file = "results/fit_BM_with_age-sex.RDS")
