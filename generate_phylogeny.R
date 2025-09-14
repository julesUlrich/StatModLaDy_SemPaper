# =============================================================
# This script generates a phylogenetic tree from language data
# using Glottolog information. The script was adapted from 
# Guzman Naranjo and Becker's paper "Statistical bias control 
# in typology".
# =============================================================

# Load required libraries
library(ape)         # for phylogenetic trees
library(tidyverse)   # for data manipulation

# -------------------------------
# Helper Functions
# -------------------------------

# Returns the parent ID for a given language ID
get_parent <- function(.id, db = families) {
  db[db$id == .id, ]$parent_id
}

# Returns the full ancestral chain for a given language ID
get_chain <- function(.id) {
  cid <- .id
  pid <- get_parent(cid)
  chain <- c(cid, pid)
  
  while (!is.na(pid)) {
    cid <- pid
    pid <- get_parent(cid)
    if (!is.na(pid)) chain <- c(chain, pid)
  }
  
  chain[!is.na(chain)]
}

# Get language/family names for a vector of IDs
get_names <- function(.idf, db = families) {
  sapply(.idf, function(.id) db[db$id == .id, ]$name)
}

# Get number of child languages for a vector of IDs
get_num_languages <- function(.idf, db = families) {
  sapply(.idf, function(.id) db[db$id == .id, ]$child_language_count)
}

# Get number of child families for a vector of IDs
get_num_families <- function(.idf, db = families) {
  sapply(.idf, function(.id) db[db$id == .id, ]$child_family_count)
}

# Custom distance function used in phylogenetic tree construction
dist.b <- function(X) {
  m <- as.matrix(as.data.frame(X)[, -1])
  rownames(m) <- as.data.frame(X)[, 1]
  m <- cbind(1:nrow(m), m)
  
  apply(m, 1, function(r) {
    cat(r[1], "/", nrow(m), " - ", sep = "")
    r[r == 0] <- -1
    rowSums(t(t(m[, -1]) == r[-1]))
  })
}

# -------------------------------
# Main Function: Build Phylogenetic Trees
# -------------------------------

build_phylos <- function(.lfd, .var, .micro_family = FALSE, distance = FALSE) {
  .var <- enquo(.var)
  
  # Build family chains for each language
  chains <- sapply(.lfd$id, function(x) {
    print(x)
    c(get_names(get_chain(x)), "TOP__")
  })
  
  # Convert each chain to a semicolon-separated string
  chain.1 <- sapply(chains, function(x) paste(x, collapse = ";"))
  
  if (.micro_family) {
    # Remove the macro family part if only micro family is desired
    chain.1 <- sapply(chain.1, function(x) str_remove(x, "^.*?;"))
  }
  
  # Get unique family labels
  all.vals <- unique(unlist(strsplit(chain.1, ";")))
  all.vals <- all.vals[!is.na(all.vals)]
  
  # Initialize dataframe with variable column
  df.philo <- select(.lfd, !!.var)
  
  # Add binary columns for each family level
  for (col in all.vals) {
    print(col)
    df.philo[, col] <- as.integer(str_detect(chain.1, col))
  }
  
  # Remove duplicates
  df.philo <- distinct(df.philo)
  
  # Compute distance or return phylogenetic tree
  df.philo_d <- dist.b(df.philo)
  
  if (distance) {
    df.philo_d
  } else {
    as.dist(1 / df.philo_d) %>%
      hclust() %>%
      as.phylo()
  }
}

# -------------------------------
# Load and Prepare Data
# -------------------------------

# Read Glottolog data
families <- read_csv("data/languoid.csv")

# Extract relevant columns for languages and families
lang_gloto_data <- families %>%
  select(id, family_id, parent_id, name)

fam_gloto_data <- families %>%
  select(id, name)

# Combine language, micro family, and macro family names
lang_fam_gloto_data <- lang_gloto_data %>%
  left_join(fam_gloto_data, by = c("family_id" = "id"), suffix = c("_language", "_macro_family")) %>%
  left_join(fam_gloto_data, by = c("parent_id" = "id"), suffix = c("_macro_family", "_micro_family")) %>%
  rename(name_micro_family = name) %>%
  mutate(
    name_micro_family = if_else(is.na(name_micro_family), name_language, name_micro_family),
    name_macro_family = if_else(is.na(name_macro_family), name_language, name_macro_family)
  )

# -------------------------------
# Build Phylogenetic Tree for UD Data
# -------------------------------

df_all <- read_csv("data/DoReCo_data.csv")
df_all$id2 <- make.unique(df_all$glottocode)

lfd <- filter(lang_fam_gloto_data, id %in% df_all$glottocode)

all_phylo <- build_phylos(lfd, name_micro_family, .micro_family = FALSE)
write_rds(all_phylo, "data/phylo.rds")
