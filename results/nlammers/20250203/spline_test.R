# Load the splines package
library(splines)

# Generate a sequence of x values (e.g., timepoints)
x <- seq(0, 100, length.out = 101)

# Compute the natural cubic spline basis with interior knots at 27, 42, 57.
# By default, ns() returns a matrix with (length(knots) + 1) columns, i.e. 4 columns here.
B <- ns(x, knots = c(27, 42, 57))

# Print the first few rows of the basis matrix
print(head(B))

# Optionally, save the basis matrix to a CSV for later comparison.
write.csv(as.matrix(B), file = "ns_basis.csv", row.names = FALSE)