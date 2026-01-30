.DEFAULT_GOAL := help

# Install package
install:
	Rscript --vanilla -e 'devtools::install()'

# Check package (CRAN style)
check:
	Rscript --vanilla -e 'devtools::check()'

# Generate documentation
man: R/*
	Rscript --vanilla -e 'devtools::document()'

# Alternative documentation command
docs: man

# Run tests
test:
	Rscript -e 'devtools::test()'


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
	@echo "  install     - Install package"
	@echo "  check       - Run R CMD check (CRAN style)"
	@echo "  man         - Generate documentation"
	@echo "  docs        - Generate documentation (alias for man)"
	@echo "  test        - Run package tests"
	@echo "  example     - Run example analysis"
	@echo "  clean       - Remove build artifacts"
	@echo "  info        - Show package info"
	@echo "  help        - Show this help"
