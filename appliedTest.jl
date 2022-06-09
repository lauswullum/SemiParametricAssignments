
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
len(x) = (q = round.(x, digits = 4), len = round(x[2] - x[1], digits = 3))
# Load the data in
root = dirname(@__FILE__)
DF = CSV.read(root * "/exercise3data.csv", DataFrame, header = [:ID, :Y, :R, :X], skipto = 2)

print("next ££££££££££££££££££££ \n")
# A first naive estimate (A.a)
fitwithout = glm(@formula(Y ~ R), DF, Binomial(), LogitLink())
coef(fitwithout)[2]
confint(fitwithout)[2, :] |> len

print("next ££££££££££££££££££££\n")
# Naive estimate using the naive influence function (don't report)
βhat, seβhat = naiveEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βhat,seβhat ), [0.025, 0.975]) |> len

print("next ££££££££££££££££££££ \n")
# The efficient estimate (A.b)
βhatEff, seβhatEff = effEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βhatEff,seβhatEff), [0.025, 0.975]) |> len

print("next ££££££££££££££££££££\n")
# The efficient estimate (A.c)
βtilde, seβtilde =  polEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βtilde,seβtilde), [0.025, 0.975]) |> len

print("next ££££££££££££££££££££\n")
# The efficient estimate (A.c)
βmis, seβmis =  effMisEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βmis,seβmis), [0.025, 0.975]) |> len

#########################################################################
# Produce tables to latex using GT in R
#########################################################################



#xWhereY1 = filter(row -> row.Y == 1, DF)
#xWhereY0 = filter(row -> row.Y == 0, DF)
###
#violin(["0"], xWhereY0.X, label = "no outcome")
#violin!(["1"],xWhereY1.X, label = "outcome")
##
#mean(xWhereY0.R) 
#mean(xWhereY1.R)
#
#histogram(xWhereY1.R, label = "outcome", normalize = true)
#histogram!(xWhereY0.R, label = "no outcome", normalize = true)
#