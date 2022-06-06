
using DataFrames
using CSV
using Plots
using StatsPlots
using DataFramesMeta

root = dirname(dirname(@__FILE__))
DF = CSV.read(root * "/exercise3data.csv", DataFrame, header = [:ID, :Y, :R, :X], skipto = 2)

xWhereY1 = filter(row -> row.Y == 1, DF)
xWhereY0 = filter(row -> row.Y == 0, DF)

violin(["0"], xWhereY0.X, label = "no outcome")
violin!(["1"],xWhereY1.X, label = "outcome")


fitwith = glm(@formula(Y ~ R + X), DF, Binomial(), LogitLink())
fitwithout = glm(@formula(Y ~ R), DF, Binomial(), LogitLink())



