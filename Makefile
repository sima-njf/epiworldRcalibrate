# Extract package name and version from DESCRIPTION
VERSION:=$(shell Rscript -e 'x<-readLines("DESCRIPTION");cat(gsub(".+[:]\\s*", "", x[grepl("^Vers", x)]))')
PKGNAME:=$(shell Rscript -e 'x<-readLines("DESCRIPTION");cat(gsub(".+[:]\\s*", "", x[grepl("^Package", x)]))')

# Mark phony targets
.PHONY: build install check clean man test document vignettes style lint debug profile

# Main build target
${PKGNAME}_${VERSION}.tar.gz: R/* inst/* man/* vignettes/*
	$(MAKE) clean ;\
	Rscript -e 'roxygen2::roxygenize()' && \
	R CMD build .

# Build package
build: ${PKGNAME}_${VERSION}.tar.gz

# Install package
install:
	Rscript --vanilla -e 'devtools::install()'

# Check package (CRAN style)
check:
	Rscript --vanilla -e 'devtools::check()'

# Generate documentation
man: R/*
	Rscript --vanilla -e 'roxygen2::roxygenize()'

# Alternative documentation command
document: man

# Run tests
test:
	Rscript -e 'devtools::test()'

# Build vignettes
vignettes:
	Rscript -e 'devtools::build_vignettes()'

# Style code
style:
	Rscript -e 'styler::style_pkg()'

# Lint code
lint:
	Rscript -e 'lintr::lint_package()'

# Load package for development
load:
	Rscript -e 'devtools::load_all()'

# Install dependencies
deps:
	Rscript -e 'devtools::install_deps(dependencies = TRUE)'

# Debug with gdb (if you have C++ code)
debug:
	R -d gdb

# Profile with valgrind (if you have C++ code)
profile: install
	R --debugger=valgrind --debugger-args='--tool=cachegrind --cachegrind-out-file=test.cache.out'

# Development workflow
dev: deps document load test

# Release workflow
release: document test check
	@echo "Package ${PKGNAME} v${VERSION} ready for release!"

# Initialize BiLSTM models (specific to your package)
init-models:
	Rscript -e 'epiworldRcalibrate::init_bilstm_model()'

# Run example analysis
example:
	Rscript -e 'rmarkdown::render("vignettes/calibration_explanation.Rmd")'

# Clean build artifacts
clean:
	rm -rf src/*.o; rm -rf src/*.a; rm -f ${PKGNAME}_${VERSION}.tar.gz; \
	rm -rf ${PKGNAME}.Rcheck/; \
	rm -rf inst/doc/

# Show package info
info:
	@echo "Package: ${PKGNAME}"
	@echo "Version: ${VERSION}"
	@echo "Tarball: ${PKGNAME}_${VERSION}.tar.gz"

# Help target
help:
	@echo "Available targets:"
	@echo "  build       - Build package tarball"
	@echo "  install     - Install package"
	@echo "  check       - Run R CMD check --as-cran"
	@echo "  check-quick - Run R CMD check (faster)"
	@echo "  test        - Run package tests"
	@echo "  document    - Generate documentation"
	@echo "  vignettes   - Build vignettes"
	@echo "  style       - Style code with styler"
	@echo "  lint        - Lint code with lintr"
	@echo "  load        - Load package for development"
	@echo "  deps        - Install dependencies"
	@echo "  dev         - Development workflow"
	@echo "  release     - Release workflow"
	@echo "  init-models - Initialize BiLSTM models"
	@echo "  example     - Run example analysis"
	@echo "  clean       - Remove build artifacts"
	@echo "  info        - Show package info"
	@echo "  help        - Show this help"
