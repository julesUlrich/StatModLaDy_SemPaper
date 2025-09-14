data {
  int N; int N_lang; array[N] int lang_id;
  int N_speaker; array[N] int speaker_id;
  int N_phoneme; array[N] int phoneme_id;
  int N_conversation; array[N] int conversation_id;
  array[N] real<lower=0> information_rate;
  matrix[N_lang, N_lang] VCV_BM;
  int prior_sim;
}

parameters {
  real p_0;
  real<lower=0> sigma_p_lang; real<lower=0> sigma_p_speaker;
  real<lower=0> sigma_p_phoneme; real<lower=0> sigma_p_conversation;
  vector[N_lang] p_lang_z; vector[N_speaker] p_speaker_z;
  vector[N_phoneme] p_phoneme_z; vector[N_conversation] p_conversation_z;
  real sigmasq_0; vector[N_phoneme] sigmasq_phoneme_z;
  real<lower=0> sigma_sigmasq_phoneme;
  vector[N_lang] phy_z; array[N_phoneme] vector[N_lang] phy_z_phoneme;
  real<lower=0> sigma_obs;
}

transformed parameters {
  vector[N_lang] p_lang_v = sigma_p_lang * p_lang_z;
  vector[N_speaker] p_speaker_v = sigma_p_speaker * p_speaker_z;
  vector[N_phoneme] p_phoneme_v = sigma_p_phoneme * p_phoneme_z;
  vector[N_conversation] p_conversation_v = sigma_p_conversation * p_conversation_z;

  vector[N_phoneme] sigmasq_phoneme;
  for (f in 1:N_phoneme)
    sigmasq_phoneme[f] = exp(sigmasq_0 + sigma_sigmasq_phoneme * sigmasq_phoneme_z[f]);

  vector[N_lang] phy_v = cholesky_decompose(exp(sigmasq_0) * VCV_BM) * phy_z;

  array[N_phoneme] vector[N_lang] phy_v_phoneme;
  for (f in 1:N_phoneme)
    phy_v_phoneme[f,] = cholesky_decompose(sigmasq_phoneme[f] * VCV_BM) * phy_z_phoneme[f];
}

model {
  p_0 ~ normal(0,1);
  sigma_p_lang ~ normal(0,1); sigma_p_speaker ~ normal(0,1);
  sigma_p_phoneme ~ normal(0,1); sigma_p_conversation ~ normal(0,1);
  p_lang_z ~ std_normal(); p_speaker_z ~ std_normal();
  p_phoneme_z ~ std_normal(); p_conversation_z ~ std_normal();
  sigmasq_0 ~ normal(0,1); sigma_sigmasq_phoneme ~ normal(0,1);
  sigmasq_phoneme_z ~ std_normal(); phy_z ~ std_normal();
  for (f in 1:N_phoneme) phy_z_phoneme[f] ~ std_normal();
  sigma_obs ~ normal(0,1) T[0,];

  if (prior_sim != 1) {
    for (i in 1:N) {
      real mu_i = p_0 + p_lang_v[lang_id[i]] + p_speaker_v[speaker_id[i]]
                + p_phoneme_v[phoneme_id[i]] + p_conversation_v[conversation_id[i]]
                + phy_v[lang_id[i]] + phy_v_phoneme[phoneme_id[i], lang_id[i]];
      target += lognormal_lpdf(information_rate[i] | mu_i, sigma_obs);
    }
  }
}

generated quantities {
  array[N] real y_rep; vector[N] log_lik;
  for (i in 1:N) {
    real mu_i = p_0 + p_lang_v[lang_id[i]] + p_speaker_v[speaker_id[i]]
              + p_phoneme_v[phoneme_id[i]] + p_conversation_v[conversation_id[i]]
              + phy_v[lang_id[i]] + phy_v_phoneme[phoneme_id[i], lang_id[i]];
    log_lik[i] = (prior_sim != 1) ? lognormal_lpdf(information_rate[i] | mu_i, sigma_obs) : -99;
    y_rep[i] = lognormal_rng(mu_i, sigma_obs);
  }
}