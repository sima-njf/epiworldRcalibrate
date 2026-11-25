# Knits the vignettes from their sources (*.Rmd.orig) into the *.Rmd files the package ships, with their output and
# figures (vignettes/figures). Building the package then runs no code, so it needs neither Python nor hours of
# model runs. Knitting runs DeepIMC (Python with torch) and the standard methods: about two hours on one core.
#
# Run from the package root, with the package installed:
#   Rscript vignettes/precompute.R                      # all three
#   Rscript vignettes/precompute.R interventions        # one
# Chunks are cached in VIGNETTE_CACHE (default vignettes/cache, not in the package), so knitting again after
# editing the text reruns only the chunks that changed.
args <- commandArgs(TRUE)
todo <- if (length(args)) args else c("calibrating-windows", "real-curves", "interventions")
old <- setwd("vignettes")
on.exit(setwd(old))
for (v in todo) {
  cat(format(Sys.time()), "knitting", v, "\n")
  knitr::knit(paste0(v, ".Rmd.orig"), paste0(v, ".Rmd"), envir = new.env(), quiet = TRUE)
}
cat(format(Sys.time()), "done\n")
