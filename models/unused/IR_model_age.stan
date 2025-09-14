data {
  int<lower=1> N;                    // Number of observations
  vector[N] y;                      // Log IR
  vector[N] x;                      // Log(SR) + Log(ER)
  int<lower=1> L;                   // Number of languages
  int<lower=1> S;                   // Number of speakers
  int<lower=1> T;                   // Number of texts
  int<lower=1> P;                   // Number of language-specific phonemes

  array[N] int<lower=1, upper=L> lang;
  array[N] int<lower=1, upper=S> speaker;
  array[N] int<lower=1, upper=T> text;
  array[N] int<lower=1, upper=P> phoneme;
  array[N] int<lower=0, upper=1> use_speaker;
  vector[N] age_scaled;
}

parameters {
  real alpha_0;

  // Non-centered parameterization
  vector[L] alpha_lang_raw;
  vector[S] beta_speaker_raw;
  vector[T] gamma_text_raw;
  vector[P] delta_phoneme_raw;

  real theta_age;

  real<lower=1e-6> sigma;
  real<lower=1e-6> sigma_lang;
  real<lower=1e-6> sigma_speaker;
  real<lower=1e-6> sigma_text;
  real<lower=1e-6> sigma_phoneme;
}

transformed parameters {
  vector[L] alpha_lang = sigma_lang * alpha_lang_raw;
  vector[S] beta_speaker = sigma_speaker * beta_speaker_raw;
  vector[T] gamma_text = sigma_text * gamma_text_raw;
  vector[P] delta_phoneme = sigma_phoneme * delta_phoneme_raw;
}

model {
  // Priors for raw effects
  alpha_lang_raw ~ normal(0, 1);
  beta_speaker_raw ~ normal(0, 1);
  gamma_text_raw ~ normal(0, 1);
  delta_phoneme_raw ~ normal(0, 1);

  // Priors on scales and fixed effects
  sigma ~ normal(0.5, 0.25);
sigma_lang ~ normal(0.5, 0.25);
sigma_speaker ~ normal(0.5, 0.25);
sigma_text ~ normal(0.5, 0.25);
sigma_phoneme ~ normal(0.5, 0.25);

  theta_age ~ normal(0, 1);
  alpha_0 ~ normal(0, 1);

  // Likelihood
  for (i in 1:N) {
    real mu = x[i] + alpha_0 +
              alpha_lang[lang[i]] +
              gamma_text[text[i]] +
              delta_phoneme[phoneme[i]] +
              theta_age * age_scaled[i];

    if (use_speaker[i] == 1)
      mu += beta_speaker[speaker[i]];

    y[i] ~ normal(mu, sigma);
  }
}
