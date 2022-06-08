
include("semiParam3.jl")

naiveStudy = makeSimStudy(naiveEst) |> x -> fromMatrixToDF(x...)
effStudy = makeSimStudy(effEst) |> x -> fromMatrixToDF(x...)
polStudy = makeSimStudy(polEst) |> x -> fromMatrixToDF(x...)
misStudy = makeSimStudy(effMisEst) |> x -> fromMatrixToDF(x...)

getMargTheo()
subset(effStudy,:type => x -> x .== "meanβ", :n => x -> x .== 400)
subset(effStudy,:type => x -> x .== "meanβ")

#subsl = subset(l2, :n => x -> x .== 200)

#########################################################################
# make Table
#########################################################################

@rput naiveStudy
@rput effStudy 
@rput polStudy

R"""
library(gt)
library(magrittr)
library(tibble)
library(tidyr)
library(dplyr)
library(reshape2)
tibble(naiveStudy) %>% 
    arrange(n)
    #select(type, ests, γ, n)
    #spread(key = "γ", value = "ests")
    #gt() %>%
    #tab_spanner(label = "n = 200", columns = c("0", "-log(4)", "-log(4)"))
"""
