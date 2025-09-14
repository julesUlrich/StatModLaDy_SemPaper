library(dplyr)
library(tibble)
library(purrr)
library(knitr)

# --- helper (same as before; keep if you need to rebuild `results`) ---
summarise_model <- function(fit, include_covariates = FALSE) {
  draws <- posterior::as_draws_df(fit)

  if (!include_covariates) {
    theta_samples <- draws$p_0
  } else {
    theta_samples <- draws$p_0 + draws$b_speaker_sex * 0 + draws$b_speaker_age * 0
  }

  intercept <- tibble::tibble(
    log_mean    = mean(theta_samples),
    log_median  = median(theta_samples),
    log_2.5     = quantile(theta_samples, 0.025),
    log_97.5    = quantile(theta_samples, 0.975),
    nat_mean    = mean(exp(theta_samples)),
    nat_2.5     = quantile(exp(theta_samples), 0.025),
    nat_97.5    = quantile(exp(theta_samples), 0.975)
  )

  out <- list(intercept = intercept)

  if (include_covariates) {
    sex <- tibble::tibble(
      mean   = mean(draws$b_speaker_sex),
      sd     = sd(draws$b_speaker_sex),
      q2.5   = quantile(draws$b_speaker_sex, 0.025),
      median = quantile(draws$b_speaker_sex, 0.5),
      q97.5  = quantile(draws$b_speaker_sex, 0.975)
    )
    age <- tibble::tibble(
      mean   = mean(draws$b_speaker_age),
      sd     = sd(draws$b_speaker_age),
      q2.5   = quantile(draws$b_speaker_age, 0.025),
      median = quantile(draws$b_speaker_age, 0.5),
      q97.5  = quantile(draws$b_speaker_age, 0.975)
    )
    out$sex <- sex
    out$age <- age
  }

  return(out)
}

# If you already have `results`, skip to the "BUILD TABLES" section.
fit_OU_no <- readr::read_rds("results/fit_OU_no_age-sex_filtered.RDS")
fit_OU_yes <- readr::read_rds("results/fit_OU_with_age-sex_filtered.RDS")
fit_BM_no <- readr::read_rds("results/fit_BM_no_age-sex_filtered.RDS")
fit_BM_yes <- readr::read_rds("results/fit_BM_with_age-sex_filtered.RDS")

results <- list(
  OU_no  = summarise_model(fit_OU_no, include_covariates = FALSE),
  OU_yes = summarise_model(fit_OU_yes, include_covariates = TRUE),
  BM_no  = summarise_model(fit_BM_no, include_covariates = FALSE),
  BM_yes = summarise_model(fit_BM_yes, include_covariates = TRUE)
)

# --- BUILD TABLES ---

# 1) Intercepts table (proper column names)
intercepts <- purrr::imap_dfr(
  results,
  ~ dplyr::bind_cols(model = .y, .x$intercept)
)

knitr::kable(
  intercepts,
  format = "latex",
  digits = 3,
  caption = "Posterior summaries of the long-term optimum (intercept) on log and natural scales."
)


# 2) Covariates table (only for models with sex/age)
covariates <- purrr::imap_dfr(results, function(x, nm) {
  if (!("sex" %in% names(x))) {
    return(NULL)
  }
  dplyr::bind_rows(
    dplyr::bind_cols(model = nm, term = "sex", x$sex),
    dplyr::bind_cols(model = nm, term = "age", x$age)
  )
})

if (nrow(covariates) > 0) {
  knitr::kable(
    covariates,
    format = "latex",
    digits = 3,
    caption = "Posterior summaries of covariate effects (log scale)."
  )
}
