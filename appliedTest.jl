
include("semiParam3.jl")


#########################################################################
# APPLIED STUDY
#########################################################################
len(x) = (q = x, len = x[2] - x[1])
# Load the data in
root = dirname(@__FILE__)
DF = CSV.read(root * "/exercise3data.csv", DataFrame, header = [:ID, :Y, :R, :X], skipto = 2)

# A first naive estimate (A.a)
fitwithout = glm(@formula(Y ~ R), DF, Binomial(), LogitLink())
coef(fitwithout)[2]
confint(fitwithout)[2, :] |> len

# Naive estimate using the naive influence function (don't report)
βhat, seβhat = naiveEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βhat,seβhat ), [0.025, 0.975]) |> len

# The efficient estimate (A.b)
βhatEff, seβhatEff = effEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βhatEff,seβhatEff), [0.025, 0.975]) |> len

#
# The efficient estimate (A.c)
βtilde, seβtilde =  polEst(DF.R, DF.Y, DF.X)
quantile.(Normal(βtilde,seβtilde), [0.025, 0.975]) |> len


#xWhereY1 = filter(row -> row.Y == 1, DF)
#xWhereY0 = filter(row -> row.Y == 0, DF)
##
#violin(["0"], xWhereY0.X, label = "no outcome")
#violin!(["1"],xWhereY1.X, label = "outcome")
#
#