# R packages for generation of table
R"""
library(gt)
library(magrittr)
library(tibble)
library(tidyr)
library(dplyr)
library(reshape2)
"""
include("semiParam3.jl")

# Run simulationstudy and transform into longformat dataframe
naiveStudy = makeSimStudy(naiveEst) |> x -> fromMatrixToDF(x...)
effStudy = makeSimStudy(effEst) |> x -> fromMatrixToDF(x...)
polStudy = makeSimStudy(polEst) |> x -> fromMatrixToDF(x...)
misStudy = makeSimStudy(effMisEst) |> x -> fromMatrixToDF(x...)

# Add a column representing the estimator
naiveStudy[!, :ex] .= "naive"
effStudy[!, :ex] .= "efficient"
polStudy[!, :ex] .= "polynomial"
misStudy[!, :ex] .= "misspecified"

# Stack the dataframes to a full dataframe 72x5
fullDF = [naiveStudy; effStudy; polStudy; misStudy]

# Compute the theoretical values of the marginal effects
getMargTheo()

#########################################################################
# Make table in R using GT pakken. 
#########################################################################

# Get path of current file
root = dirname(@__FILE__)

# Put the full dataframe and path to R using RCall.jl
@rput fullDF
@rput root

# Make table and write a latexfile out. 
R"""
tibble(fullDF) %>% 
    unite("n,γ", n:γ) %>%
    spread(key = "n,γ", value = "ests") %>%
    gt(
        rowname_col = "type", 
        groupname_col = "ex"
    ) %>%
    tab_spanner(label = "n = 200", columns = c("200_0", "200_-log(4)", "200_-log(6)")) %>%
    tab_spanner(label = "n = 400", columns = c("400_0", "400_-log(4)", "400_-log(6)")) %>%
    cols_label("400_0" = "0", "400_-log(4)" = "-log(4)", "400_-log(6)" = "-log(6)", 
                "200_0" = "0", "200_-log(4)" = "-log(4)", "200_-log(6)" = "-log(6)"
    ) %>%
    fmt_number(columns = 1:8, decimals = 3) %>%
    tab_stubhead(label = "γ") %>%
    as_latex() 
    as.character() %>%
    writeLines(., con = paste(root, "/latex_table.tex", sep = ""))
"""



