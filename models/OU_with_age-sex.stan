data{
  int N;                              // observations
  int N_lang;                         // languages
  array[N] int lang_id;               // 1..N_lang

  int N_speaker;                      // speakers
  array[N] int speaker_id;            // 1..N_speaker

  int N_phoneme;                      // phonemes
  array[N] int phoneme_id;            // 1..N_phoneme

  int N_conversation;                 // conversations
  array[N] int conversation_id;       // 1..N_conversation

  // speaker-level covariates (centered/scaled)
  vector[N_speaker] sex_s;   // e.g., female=-0.5, male=+0.5
  vector[N_speaker] age_s;   // centered and scaled
  
  // Outcome (positive, continuous)
  array[N] real<lower=0> information_rate;

  // OU needs patristic distance matrix in the SAME order as lang_id indexing
  matrix[N_lang, N_lang] dist_mat;

  int prior_sim; // 1 = prior predictive only (no data likelihood)
}

parameters{
// Fixed effects on log scale (lognormal mean)
  real p_0;               // intercept
  
  // Speaker intercept model
  real a_speaker;          // mean speaker intercept
  real b_speaker_sex;      // effect of sex on speaker intercept
  real b_speaker_age;      // effect of age on speaker intercept
  real<lower=0> sigma_p_speaker;
  
   // Random intercept SDs
  real<lower=0> sigma_p_lang;
  real<lower=0> sigma_p_phoneme;
  real<lower=0> sigma_p_conversation;

  // Random intercept standard-normal draws
  vector[N_lang]    p_lang_z;
  vector[N_speaker] p_speaker_z;
  vector[N_phoneme] p_phoneme_z;
  vector[N_conversation] p_conversation_z;
  
  real eta_0;                        // Global intercept for OU cov: variance (sigma)
  real alpha_0;                      // Global intercepts for OU cov: alpha 
  
  // Per-phoneme eta deviations 
  vector[N_phoneme] eta_phoneme_z;
  real<lower=0> sigma_eta_phoneme;
  
  // Per-phoneme alpha deviations 
  vector[N_phoneme] alpha_phoneme_z; 
  real<lower=0> sigma_alpha_phoneme; 
  
  vector[N_lang] phy_z;                          // global OU effect
  array[N_phoneme] vector[N_lang] phy_z_phoneme; // per-phoneme OU effect

  // Observation (residual) SD on log scale
  real<lower=0> sigma_obs;
}

transformed parameters{
  // Random intercepts (scaled)
  vector[N_lang]    p_lang_v        = sigma_p_lang        * p_lang_z;
  vector[N_phoneme] p_phoneme_v     = sigma_p_phoneme     * p_phoneme_z;
  vector[N_conversation] p_conversation_v = sigma_p_conversation * p_conversation_z;
  vector[N_lang] phy_v;
  
  // speaker intercepts include sex/age at the speaker level
  vector[N_speaker] p_speaker_v =
    a_speaker
  + b_speaker_sex * sex_s
  + b_speaker_age * age_s
  + sigma_p_speaker * p_speaker_z;
  
  // OU per-phoneme variance (alpha, eta on original scale)
  vector[N_phoneme] alpha_phoneme;
  vector[N_phoneme] eta_phoneme;
  for (p in 1:N_phoneme) {
    alpha_phoneme[p] = exp(alpha_0 + alpha_phoneme_z[p] * sigma_alpha_phoneme);
    eta_phoneme[p] = exp(eta_0 + sigma_eta_phoneme * eta_phoneme_z[p]);
  }


  /* Global phylogenetic pars */
  {
    matrix[N_lang,N_lang] phy_cov;
    for (i in 1:(N_lang-1))
      for (j in (i+1):N_lang) {
        phy_cov[i,j] = exp(eta_0) * exp( -(exp(alpha_0) * dist_mat[i,j]) );
        phy_cov[j,i] = phy_cov[i,j];
      }
      for (q in 1:N_lang) phy_cov[q,q] = exp(eta_0) + 0.001;
      
      phy_v = cholesky_decompose(phy_cov) * phy_z;
  }

  // Per-phoneme OU covariance, varying only the variance (eta_phoneme[p])
  // (keeps the same alpha across phonemes, mirroring BM's per-phoneme variance only)
  array[N_phoneme] vector[N_lang] phy_v_phoneme;
  {
    for (p in 1:N_phoneme) {
      matrix[N_lang, N_lang] phy_cov_p;
      real eta_p = eta_phoneme[p];
      real alpha_p = alpha_phoneme[p];
      for (i in 1:N_lang) {
        phy_cov_p[i,i] = eta_p + 0.001;
        for (j in (i+1):N_lang) {
          real v = eta_p * exp( -alpha_p * dist_mat[i,j] );
          phy_cov_p[i,j] = v;
          phy_cov_p[j,i] = v;
        }
      }
      phy_v_phoneme[p,] = cholesky_decompose(phy_cov_p) * phy_z_phoneme[p];
    }
    
  }
}

model{
  // Priors
  p_0 ~ normal(0, 1);

  a_speaker ~ normal(0, 1);
  b_speaker_sex ~ normal(0, 1);
  b_speaker_age ~ normal(0, 1);
  
  sigma_p_lang        ~ normal(0, 1);
  sigma_p_speaker     ~ normal(0, 1);
  sigma_p_phoneme     ~ normal(0, 1);
  sigma_p_conversation~ normal(0, 1);

  p_lang_z        ~ std_normal();
  p_speaker_z     ~ std_normal();
  p_phoneme_z     ~ std_normal();
  p_conversation_z~ std_normal();

  eta_0   ~ normal(0, 1);   // log-variance
  alpha_0 ~ normal(0, 1);   // log-alpha

  sigma_eta_phoneme ~ normal(0, 1);
  eta_phoneme_z     ~ std_normal();
  sigma_alpha_phoneme ~ normal(0, 1);
  alpha_phoneme_z     ~ std_normal();
  
  phy_z ~ std_normal();
  for (f in 1:N_phoneme) phy_z_phoneme[f] ~ std_normal();

  sigma_obs ~ normal(0, 1) T[0,];
  
  // Likelihood (lognormal)
  if (prior_sim != 1) {
    for (i in 1:N) {
      real mu_i = p_0
                + p_lang_v[lang_id[i]]
                + p_speaker_v[speaker_id[i]]
                + p_phoneme_v[phoneme_id[i]]
                + p_conversation_v[conversation_id[i]]
                + phy_v[lang_id[i]]
                + phy_v_phoneme[phoneme_id[i], lang_id[i]];
      target += lognormal_lpdf(information_rate[i] | mu_i, sigma_obs);
    }
  }
}

generated quantities{
  array[N] real y_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    real mu_i = p_0
              + p_lang_v[lang_id[i]]
              + p_speaker_v[speaker_id[i]]
              + p_phoneme_v[phoneme_id[i]]
              + p_conversation_v[conversation_id[i]]
              + phy_v[lang_id[i]]
              + phy_v_phoneme[phoneme_id[i], lang_id[i]];
    if (prior_sim != 1) {
      log_lik[i] = lognormal_lpdf(information_rate[i] | mu_i, sigma_obs);
    } else {
      log_lik[i] = -99;
    }
    y_rep[i] = lognormal_rng(mu_i, sigma_obs);
  }
}
