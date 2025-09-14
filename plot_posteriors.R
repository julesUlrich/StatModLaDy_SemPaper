library(tidyr)
library(ggplot2)
library(posterior)
library(dplyr)

fit_OU_no <- readr::read_rds("results/fit_OU_no_age-sex_filtered.RDS")
fit_OU_yes <- readr::read_rds("results/fit_OU_with_age-sex_filtered.RDS")
fit_BM_no <- readr::read_rds("results/fit_BM_no_age-sex_filtered.RDS")
fit_BM_yes <- readr::read_rds("results/fit_BM_with_age-sex_filtered.RDS")

# Combine draws
draws_all <- bind_rows(
  as_draws_df(fit_OU_no) %>% mutate(model = "OU_no"),
  as_draws_df(fit_OU_yes) %>% mutate(model = "OU_yes"),
  as_draws_df(fit_BM_no) %>% mutate(model = "BM_no"),
  as_draws_df(fit_BM_yes) %>% mutate(model = "BM_yes")
)

tidy_draws <- draws_all %>%
  pivot_longer(
    cols = starts_with("p_0") | starts_with("b_speaker"),
    names_to = "parameter", values_to = "value"
  )

# Custom color palette
model_colors <- c(
  "OU_no" = "#1b9e77", "OU_yes" = "#d95f02",
  "BM_no" = "#7570b3", "BM_yes" = "#e7298a"
)

p1 <- ggplot(tidy_draws, aes(x = value, y = model, color = model)) +
  stat_halfeye(.width = c(0.5, 0.95)) +
  facet_wrap(~parameter, scales = "free_x") +
  scale_color_manual(values = model_colors) +
  theme_minimal() +
  ggtitle("Posterior distributions of key parameters across models")

# Save
ggsave("plots/posterior_parameters.png", p1, width = 8, height = 6, dpi = 300)

# Density of long-term optima
theta_df <- bind_rows(
  data.frame(theta = exp(as_draws_df(fit_OU_no)$p_0), model = "OU_no"),
  data.frame(theta = exp(as_draws_df(fit_OU_yes)$p_0), model = "OU_yes"),
  data.frame(theta = exp(as_draws_df(fit_BM_no)$p_0), model = "BM_no"),
  data.frame(theta = exp(as_draws_df(fit_BM_yes)$p_0), model = "BM_yes")
)

p2 <- ggplot(theta_df, aes(x = theta, fill = model)) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = model_colors) +
  scale_x_continuous("Long-term optimum (natural scale)") +
  theme_minimal() +
  ggtitle("Posterior distributions of long-term optima")

ggsave("plots/posterior_optima.png", p2, width = 8, height = 6, dpi = 300)

# Covariate plots
p3 <- ggplot(data.frame(sex = as_draws_df(fit_OU_yes)$b_speaker_sex), aes(x = sex)) +
  geom_density(fill = "#1f78b4", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal() +
  xlab("Effect of Sex on log(Information Rate)") +
  ggtitle("Posterior distribution of sex effect (OU model)")

ggsave("plots/posterior_sex_effect.png", p3, width = 8, height = 6, dpi = 300)

p4 <- ggplot(data.frame(age = as_draws_df(fit_OU_yes)$b_speaker_age), aes(x = age)) +
  geom_density(fill = "#33a02c", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal() +
  xlab("Effect of Age on log(Information Rate)") +
  ggtitle("Posterior distribution of age effect (OU model)")

ggsave("plots/posterior_age_effect.png", p4, width = 8, height = 6, dpi = 300)
