using RCall

# R packages for generation of table using GT
R"""
library(gt)
library(magrittr)
library(tibble)
library(tidyr)
library(dplyr)
library(reshape2)
"""
include("semiParam3.jl")


#########################################################################
# APPLIED STUDY
#########################################################################
# Load the data in
root = dirname(@__FILE__)
DF = CSV.read(root * "/exercise3data.csv", DataFrame, header = [:ID, :Y, :R, :X], skipto = 2)


#########################################################################
# Produce dataframe
#########################################################################
# Do a GLM fit of the naive estimator
fitwithout = glm(@formula(Y ~ R), DF, Binomial(), LogitLink())
coefGLM = coef(fitwithout)[2]
confintGLMLower, confintGLMUpper = confint(fitwithout)[2, :]

# Compute the efficient estimate
βhatEff, seβhatEff = effEst(DF.R, DF.Y, DF.X)
confintHatEffLower, confintHatEffUpper = quantile.(Normal(βhatEff,seβhatEff), [0.025, 0.975])

# Compute the polynomial estimator
βtilde, seβtilde =  polEst(DF.R, DF.Y, DF.X)
confintPolLower, confintPolUpper =  quantile.(Normal(βtilde,seβtilde), [0.025, 0.975]) 

# Collect estimates into dataframe
DFest = DataFrame(
    type = ["Naive", "Efficient", "Polynomial"],
    estimate = [coefGLM, βhatEff, βtilde], 
    lower = [confintGLMLower, confintHatEffLower,confintPolLower],
    upper = [confintGLMUpper, confintHatEffUpper,confintPolUpper]
)

# get odds ratios
#exp.(select(DFest, [:estimate, :lower, :upper]))


#########################################################################
# Produce tables to latex using GT in R
#########################################################################

# Get path of current file
root = dirname(@__FILE__)

# Put the full dataframe and path to R using RCall.jl
@rput DFest
@rput root

# Output the latex file with table code
R"""
tibble(DFest) |>
    gt() |>
    fmt_number(
        columns = 2:4,
        decimals = 3
    ) |>
    tab_spanner(label = "confidence interval", columns = c("lower", "upper")        
    ) |>
    as_latex() |>
    as.character() |>
    writeLines(con = paste(root, "/latexApplied.tex", sep = ""))
"""

# Print table
pretty_table(DFest, formatters = ft_printf("%5.3f"), alignment = :c)