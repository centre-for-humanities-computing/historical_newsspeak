# requirements.R
# Install all required packages for the analysis

required_packages <- c(
  "mgcv",        # Generalized Additive Models
  "gratia",      # Diagnostics for GAMs
  "dplyr",       # Data manipulation
  "tidyr",       # Data tidying
  "arrow",       # Fast data import/export
  "ggplot2",     # Visualization
  "patchwork",   # Multi-panel plots
  "tibble",      # Modern data frames
  "here"         # File path management
)

# Install missing packages
missing_packages <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

# Set default seed
SEED <- 20260811
set.seed(SEED)