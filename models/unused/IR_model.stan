data {
  int<lower=1> N;                       // Number of observations
  vector[N] y;                          // log(IR)
  vector[N] x;                          // log(SR) + log(ER)

  int<lower=1> L;                       // Number of languages
  int<lower=1> S;                       // Number of speakers
  int<lower=1> T;                       // Number of texts
  int<lower=1> P;                       // Number of phonemes

  array[N] int<lower=1, upper=L> lang;
  array[N] int<lower=1, upper=S> speaker;
  array[N] int<lower=1, upper=T> text;
  array[N] int<lower=1, upper=P> phoneme;

  array[N] int<lower=0, upper=1> use_speaker;  // whether to include speaker effect
  array[N] int<lower=1> freq;                  // frequency of this phoneme in the text
}

parameters {
  real alpha_0;

  vector[L] z_alpha_lang;
  vector[S] z_beta_speaker;
  vector[T] z_gamma_text;
  vector[P] z_delta_phoneme;

  real<lower=0> sigma;
  real<lower=0> sigma_lang;
  real<lower=0> sigma_speaker;
  real<lower=0> sigma_text;
  real<lower=0> sigma_phoneme;
}

transformed parameters {
  vector[L] alpha_lang = z_alpha_lang * sigma_lang;
  vector[S] beta_speaker = z_beta_speaker * sigma_speaker;
  vector[T] gamma_text = z_gamma_text * sigma_text;
  vector[P] delta_phoneme = z_delta_phoneme * sigma_phoneme;
}

model {
  // Priors
  alpha_0 ~ normal(0, 1);

  z_alpha_lang ~ normal(0, 1);
  z_beta_speaker ~ normal(0, 1);
  z_gamma_text ~ normal(0, 1);
  z_delta_phoneme ~ normal(0, 1);

  sigma ~ normal(0, 1);
  sigma_lang ~ cauchy(0, 2.5);
  sigma_speaker ~ normal(0, 1);
  sigma_text ~ normal(0, 1);
  sigma_phoneme ~ normal(0, 1);

  // Likelihood (weighted by frequency)
  for (i in 1:N) {
    real mu = x[i]
              + alpha_0
              + alpha_lang[lang[i]]
              + gamma_text[text[i]]
              + delta_phoneme[phoneme[i]];

    if (use_speaker[i] == 1)
      mu += beta_speaker[speaker[i]];

    // frequency-weighted log-likelihood
    target += freq[i] * normal_lpdf(y[i] | mu, sigma);
  }
}
