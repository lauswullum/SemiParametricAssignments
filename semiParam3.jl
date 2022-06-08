using CSV
using GLM
using RCall
using Plots
using QuadGK
using DataFrames
using StatsFuns
using StatsPlots
using Distributions
using BenchmarkTools
using DataFramesMeta


#########################################################################
# IMPLEMENTATIONS FOR SIMULATION STUDY
#########################################################################



function expit(x)
    1 / (1 + exp(-x))
end

# Variance of mean - almost second derivative
# Used in sandwich estimation
function expit2(x)
    exp(x) / (1 + exp(x))^2
end


# Simulation of data from binary outcome RCT
# with true values of α = -0.5 and β = 0.3
function simData(n, γ)
    R = rand(Bernoulli(0.5), n)
    X = randn(n)
    condMean = -0.5 .+ 0.3*R + γ*X
    μ = expit.(condMean)
    y = @. rand(Bernoulli(μ))
    return (R=R, Y=y, X = X)
end

# derivative of g wrt. p0
function gprime0(p0)
    1 / (p0*(p0 - 1))
end

# derivative of g wrt. p1
function gprime1(p1)
    1 / (p1*(1 - p1))
end

# Naive estimator and Variance
function naiveEst(R, Y, X)
    n = length(R)
    estVec0 = zeros(n)
    estVec1 = zeros(n)  
    p0hat = sum((1 .- R) .* Y) / sum(1 .- R)
    p1hat = sum(R .* Y) / sum(R)
    βhat = log(p1hat / (1 - p1hat)) - log(p0hat / (1 - p0hat))
    δhat = mean(R)
    influenceVec = [(1 - R[i])*(Y[i] - p0hat) * gprime0(p0hat) / (1 - δhat) + 
                    R[i] * (Y[i] - p1hat) * gprime1(p1hat) / δhat for i in 1:n]
    estSD = std(influenceVec)
    (βhat = βhat, ϕϕhat = estSD/sqrt(n))
end

#print("what \n")
#R, Y = simData(1000, 0)
#std([first(naiveEst(simData(1000, 0)...)) for i in 1:5000])|> print
#print("\n")
#naiveEst(R, Y)[2] |> print

function effEst(R,Y,X)
    n = length(R)
    p0hat = sum((1 .- R) .* Y) / sum(1 .- R)
    p1hat = sum(R .* Y) / sum(R)
    δhat = mean(R)
    βhat = log(p1hat / (1 - p1hat)) - log(p0hat / (1 - p0hat))
    fit = glm(@formula(Y ~ R + X), (;R, Y, X ), Binomial(), LogitLink())
    pred0 = predict(fit, (;R = zeros(Int, size(R)), X = X ))
    pred1 = predict(fit, (;R = ones(Int, size(R)), X = X ))
    influenceVec = zeros(n)
    for i in 1:n
        influenceVec[i] += gprime0(p0hat) * (1 - R[i]) * (Y[i] - p0hat) / (1-δhat)
        influenceVec[i] += gprime1(p1hat) * R[i] * (Y[i] - p1hat) / δhat
        influenceVec[i] += gprime0(p0hat) * (R[i] - δhat) * (pred0[i] - p0hat) / (1-δhat)
        influenceVec[i] -= gprime1(p1hat) * (R[i] - δhat) * (pred1[i] - p1hat) / δhat
    end
    βhatEff = βhat + mean(influenceVec)
    estSD = std(influenceVec)
    (βhat = βhatEff, ϕϕhat = estSD/sqrt(n))
end

#effEst(DF.R, DF.Y, DF.X)

function effMisEst(R,Y,X)
    n = length(R)
    p0hat = sum((1 .- R) .* Y) / sum(1 .- R)
    p1hat = sum(R .* Y) / sum(R)
    δhat = mean(R)
    βhat = log(p1hat / (1 - p1hat)) - log(p0hat / (1 - p0hat))
    fit = lm(@formula(Y ~ R + X), (;R, Y, X ))
    pred0 = predict(fit, (;R = zeros(Int, size(R)), X = X ))
    pred1 = predict(fit, (;R = ones(Int, size(R)), X = X ))
    influenceVec = zeros(n)
    for i in 1:n
        influenceVec[i] += gprime0(p0hat) * (1 - R[i]) * (Y[i] - p0hat) / (1-δhat)
        influenceVec[i] += gprime1(p1hat) * R[i] * (Y[i] - p1hat) / δhat
        influenceVec[i] += gprime0(p0hat) * (R[i] - δhat) * (pred0[i] - p0hat) / (1-δhat)
        influenceVec[i] -= gprime1(p1hat) * (R[i] - δhat) * (pred1[i] - p1hat) / δhat
    end
    βhatEff = βhat + mean(influenceVec)
    estSD = std(influenceVec)
    (βhat = βhatEff, ϕϕhat = estSD/sqrt(n))
end

function qExp(X)
    [ones(Float64, size(X))  X  X.^2 X.^3]'
end

function polEst(R,Y,X)
    n = length(R)
    p0hat = sum((1 .- R) .* Y) / sum(1 .- R)
    p1hat = sum(R .* Y) / sum(R)
    δhat = mean(R)
    βhat = log(p1hat / (1 - p1hat)) - log(p0hat / (1 - p0hat))
    influenceVec = [(1 - R[i])*(Y[i] - p0hat) * gprime0(p0hat) / (1 - δhat) + 
                    R[i] * (Y[i] - p1hat) * gprime1(p1hat) / δhat for i in 1:n]
    # Compute θ₀
    θ0hat1 =  inv(qExp(X) * qExp(X)')
    θ0hat2 = qExp(X) * ((R .- δhat) .* influenceVec)
    θhat0 = 1 / (δhat*(1 - δhat)) *  θ0hat1' * θ0hat2
    ϕtilde = influenceVec .- ((θhat0' * qExp(X))' .* (R .- δhat))
    βtilde = βhat + mean(ϕtilde)
    estSD = std(ϕtilde)
    (βtilde = βtilde, ϕϕhat = estSD/sqrt(n))
end


#R,Y,X = simData(200, 0)
#polEst(R,Y,X)

#histogram([first(naiveEst(simData(400, 0)...)) for i in 1:2000], label = "non-eff")
#histogram!([first(effEst(simData(400, 0)...)) for i in 1:2000], label = "eff")
#histogram!([polBasisEst(simData(400, 0))...) for i in 1:2000], label = "pol-eff")

function logORMargTheo()
    α = -0.5
    β = 0.3 
    σX = 1
    function logORZ(γ)
        # Get normal pdf
        normalpdf(x) = pdf(Normal(0, σX), x)
        #
        pY1R1(x) = normalpdf(x) * expit(α + β + γ*x)
        #pY0R1(x) = normalPdf(x) * expitAlt(β + γ*x)
        # and
        pY1R0(x) = normalpdf(x) * expit(α + γ*x)
        #pY0R0(x) = normalPdf(x) * expitAlt(γ*x)
        #then
        # Numerator is 
        numpY1R1 = first(quadgk(pY1R1, -Inf, Inf, rtol = 1e-3))
        numpY0R1 = 1 - numpY1R1 
        # Denominator is
        denumpY1R0 = first(quadgk(pY1R0, -Inf, Inf, rtol = 1e-3))
        denumpY0R0 = 1 - denumpY1R0
        # finally
        #log((numpY1R1 / numpY0R1) / (denumpY1R0 / denumpY0R0))
        log(numpY1R1) - log(numpY0R1) - (log(denumpY1R0) - log(denumpY0R0))
    end
    return logORZ
end

# Compute thoeretical marginal log OR for each γ in the param:γlist
function computeTheoMargLogOr(γlist)
    saveTheoMargLogOr = Array{Float64}(undef,length(γlist))
    # Compute the thoeretical marginal log odds
    for i in eachindex(γlist)
        logORmarg = logORMargTheo()
        saveTheoMargLogOr[i] = logORmarg(γlist[i])
    end
    saveTheoMargLogOr
end

function getMargTheo()
    γlist = [0, -log(4), -log(6)]
    computeTheoMargLogOr(γlist)
end

# compute one run of the simulation study
function oneSimRun(n, γ, estimator)
    betaVec = zeros(2000)
    betaSDVec = zeros(2000)
    @inbounds for i in 1:2000
        R, Y, X = simData(n, γ)
        betaVec[i], betaSDVec[i] = estimator(R,Y,X)
    end
    return (meanβ = mean(betaVec), sdβ = std(betaVec), meanSDβ = mean(betaSDVec))
end


#oneSimRun(200, 0, naiveEst)
#oneSimRun(200, -log(4), effEst)

function makeSimStudy(estimator)
    nlist = [200, 400]
    γlist = [0, -log(4), -log(6)]
    # γ x n matrix 
    saveMeanβ   = Array{Float64}(undef, 3, 2)
    saveSDβ     = Array{Float64}(undef, 3, 2)
    saveMeanSDβ = Array{Float64}(undef, 3, 2)
    for i in eachindex(γlist)
        for j in eachindex(nlist)
            a,b,c = [oneSimRun(nlist[j], γlist[i], estimator)...]
            saveMeanβ[i,j]   = a
            saveSDβ[i,j]     = b
            saveMeanSDβ[i,j] = c
        end
    end
    return (saveMeanβ, saveSDβ, saveMeanSDβ)
end

function fromMatrixToDF(saveMeanβ, saveSDβ, saveMeanSDβ)
    estimate = ["meanβ", "sdβ", "meanSDβ"]
    nlistNext = [200, 200, 200, 400, 400, 400]
    γlist = ["0", "-log(4)", "-log(6)"]
    df1 = DataFrame(
        ests = saveMeanβ[:],
        type = estimate[1], 
        n = nlistNext, 
        γ = repeat(γlist, 2)
        )
    df2 = DataFrame(
        ests = saveSDβ[:],
        type = estimate[2], 
        n = nlistNext, 
        γ = repeat(γlist, 2)
        )
    df3 = DataFrame(
        ests = saveMeanSDβ[:],
        type = estimate[3], 
        n = nlistNext, 
        γ = repeat(γlist, 2)
        )
    [df1 ; df2 ; df3]
end






