# R packages for generation of table
using RCall
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
num = 20000
naiveStudy = makeSimStudy(naiveEst, num) |> x -> fromMatrixToDF(x...)
effStudy   = makeSimStudy(effEst, num) |> x -> fromMatrixToDF(x...)
polStudy   = makeSimStudy(polEst, num) |> x -> fromMatrixToDF(x...)
misStudy   = makeSimStudy(effMisEst, num) |> x -> fromMatrixToDF(x...)

# A small function to do pretty printing. 
function flotPrint(datenGiven)
    daten = copy(datenGiven)
    daten.nγ = string.("n=", daten.n, ",γ=", daten.γ)
    select!(daten, [:ests, :type, :nγ])
    daten = unstack(daten, :type, :nγ, :ests)
    pretty_table(daten, formatters = ft_printf("%5.3f"), alignment = :c, nosubheader = true)
end

# Print tables of simulation study.
print(Panel("naive estimator", width = 30))
flotPrint(naiveStudy)
print(Panel("efficient estimator", width = 30))
flotPrint(effStudy)
print(Panel("polynomial estimator", width = 30))
flotPrint(polStudy)
print(Panel("misspecified estimator", width = 30))
flotPrint(misStudy)

# Add a column representing the estimator type
naiveStudy[!, :ex] .= "naive"
effStudy[!, :ex] .= "efficient"
polStudy[!, :ex] .= "polynomial"
misStudy[!, :ex] .= "misspecified"


# Stack the dataframes to a full dataframe 72x5
fullDF = [naiveStudy; effStudy; polStudy; misStudy]

# Compute the theoretical values of the marginal effects
print("\n \n")
print("True marginal log odds ratio for the simulation study\n")
print("γ = 0, γ = -log(4), γ = -log(6)\n")
print(getMargTheo())

#########################################################################
# Make table in R using GT pakken. 
#########################################################################

# Get path of current file
root = dirname(@__FILE__)

# Put the full dataframe and path to R using RCall.jl
@rput fullDF
@rput root

# Make table and write a latexfile in the current directory. 
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
    as_latex() %>%
    as.character() %>%
    writeLines(., con = paste(root, "/latexSimstudy.tex", sep = ""))
"""

