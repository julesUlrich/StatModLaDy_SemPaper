data {
  int N; int N_lang; array[N] int lang_id;
  int N_speaker; array[N] int speaker_id;
  int N_phoneme; array[N] int phoneme_id;
  int N_conversation; array[N] int conversation_id;
  array[N] real<lower=0> information_rate;
  matrix[N_lang,N_lang] dist_mat; int prior_sim;
}

parameters {
  real p_0; real<lower=0> sigma_p_lang, sigma_p_speaker, sigma_p_phoneme, sigma_p_conversation;
  vector[N_lang] p_lang_z; vector[N_speaker] p_speaker_z;
  vector[N_phoneme] p_phoneme_z; vector[N_conversation] p_conversation_z;
  real eta_0, alpha_0; vector[N_phoneme] eta_phoneme_z, alpha_phoneme_z;
  real<lower=0> sigma_eta_phoneme, sigma_alpha_phoneme;
  vector[N_lang] phy_z; array[N_phoneme] vector[N_lang] phy_z_phoneme;
  real<lower=0> sigma_obs;
}

transformed parameters {
  vector[N_lang] p_lang_v = sigma_p_lang * p_lang_z;
  vector[N_speaker] p_speaker_v = sigma_p_speaker * p_speaker_z;
  vector[N_phoneme] p_phoneme_v = sigma_p_phoneme * p_phoneme_z;
  vector[N_conversation] p_conversation_v = sigma_p_conversation * p_conversation_z;

  vector[N_phoneme] alpha_phoneme, eta_phoneme;
  for (p in 1:N_phoneme) {
    alpha_phoneme[p] = exp(alpha_0 + alpha_phoneme_z[p] * sigma_alpha_phoneme);
    eta_phoneme[p] = exp(eta_0 + sigma_eta_phoneme * eta_phoneme_z[p]);
  }

  matrix[N_lang,N_lang] phy_cov;
  for (i in 1:N_lang) for (j in 1:N_lang)
    phy_cov[i,j] = (i==j) ? exp(eta_0)+0.001 : exp(eta_0)*exp(-exp(alpha_0)*dist_mat[i,j]);
  vector[N_lang] phy_v = cholesky_decompose(phy_cov)*phy_z;

  array[N_phoneme] vector[N_lang] phy_v_phoneme;
  for (p in 1:N_phoneme) {
    matrix[N_lang,N_lang] phy_cov_p;
    for (i in 1:N_lang) for (j in 1:N_lang)
      phy_cov_p[i,j] = (i==j) ? eta_phoneme[p]+0.001 : eta_phoneme[p]*exp(-alpha_phoneme[p]*dist_mat[i,j]);
    phy_v_phoneme[p,] = cholesky_decompose(phy_cov_p)*phy_z_phoneme[p];
  }
}

model {
  p_0 ~ normal(0,1); sigma_p_lang ~ normal(0,1); sigma_p_speaker ~ normal(0,1);
  sigma_p_phoneme ~ normal(0,1); sigma_p_conversation ~ normal(0,1);
  p_lang_z ~ std_normal(); p_speaker_z ~ std_normal(); p_phoneme_z ~ std_normal(); p_conversation_z ~ std_normal();
  eta_0 ~ normal(0,1); alpha_0 ~ normal(0,1);
  sigma_eta_phoneme ~ normal(0,1); eta_phoneme_z ~ std_normal();
  sigma_alpha_phoneme ~ normal(0,1); alpha_phoneme_z ~ std_normal();
  phy_z ~ std_normal(); for (f in 1:N_phoneme) phy_z_phoneme[f] ~ std_normal();
  sigma_obs ~ normal(0,1) T[0,];

  if (prior_sim != 1)
    for (i in 1:N) {
      real mu_i = p_0 + p_lang_v[lang_id[i]] + p_speaker_v[speaker_id[i]]
                + p_phoneme_v[phoneme_id[i]] + p_conversation_v[conversation_id[i]]
                + phy_v[lang_id[i]] + phy_v_phoneme[phoneme_id[i], lang_id[i]];
      target += lognormal_lpdf(information_rate[i] | mu_i, sigma_obs);
    }
}