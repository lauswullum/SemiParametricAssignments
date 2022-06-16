using Distributions
using GLM
using Statistics
using Plots
using StatsPlots
using Plots.PlotMeasures
using NLsolve
using LaTeXStrings
using ForwardDiff

#
# ██████ ██ ███    ███ ██    ██ ██       █████  ████████ ██  ██████  ███    ██ 
# █      ██ ████  ████ ██    ██ ██      ██   ██    ██    ██ ██    ██ ████   ██ 
# ██████ ██ ██ ████ ██ ██    ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██ 
#     ██ ██ ██  ██  ██ ██    ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██ 
# ██████ ██ ██      ██  ██████  ███████ ██   ██    ██    ██  ██████  ██   ████ 
#                                                                             
#
# ███████ ████████ ██    ██ ██████  ██    ██ 
# ██         ██    ██    ██ ██   ██  ██  ██  
# ███████    ██    ██    ██ ██   ██   ████   
#      ██    ██    ██    ██ ██   ██    ██    
# ███████    ██     ██████  ██████     ██    
#                                                                       


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
function simData(n,α, β)
    R = rand(Uniform(0,1), n)
    condMean = α .+ β * R
    μ = expit.(condMean)
    y = @. rand(Bernoulli(μ))
    return (R=R, y=y, μ=μ)
end

# GEE with wierd weight matrix
function worseEst(R, Y, guess)
    getAs(R) = [1 min(0.1, R)]
    A = hcat((getAs.(R))'...)
    function g(θ)
        A * (Y -  expit.(θ[1] .+ θ[2] * R))
    end
    # NLsolve package solves the non-linear equation
    sol = nlsolve(g, guess, autodiff = :forward, method = :newton)
    #propertynames(sol)
    # Extract the solution
    if sol.f_converged == false
        print("Not Converged")
    end
    sol.zero
end


# One-step (newton-raphson) estimation given θhat
function oneStep(R,Y, θhat)
    n = length(Y)
    getAs(R) = [1 R]
    A = hcat((getAs.(R))'...)
    function gOpt(θ)
        A * (Y -  expit.(θ[1] .+ θ[2] * R))
    end
    # Use zygote.jl to compute the exact jacobian
    #gPrimeOpt(θ) = 1/n .* jacobian(gOpt, θ) |> first

    # Use ForwardDiff.jl to compute the exact jacobian
    gJac = θ -> ForwardDiff.jacobian(gOpt, θ)
    #gPrimeOpt(θ) = 1/n .* ForwardDiff.jacobian(gOpt, θ)
    gPrimeOpt(θ) = 1/n .* gJac(θ)
    # The actual one-step estimator
    θnew = θhat - inv(gPrimeOpt(θhat)) * (1/n .* gOpt(θhat))
end

#R,Y, = simData(1000, 0, 0)
#s = worseEst(R, Y, [0.0, 0.0])
#oneStep(R,Y, s)

# Fit a logistic GLM and extract coefficients
function glmEst(R, Y)
    data = (;R, Y)
    fit  = glm(@formula(Y ~ R), data, Binomial(), LogitLink())
    coef(fit)
end


#########################################################################
# SIMULATION STUDY
#########################################################################

#(1) For sample size n = 250, 500, 1000:
#For r = 1, . . . , 1000 simulations:
#(1.1) Tweak the function developed during Exercise 1 and simulate a random data set
#(Y i , R i ) ni=1 with parameters (α, β, γ) = (0, 0, 0) .
#(1.2) Get the initial estimator β̂ from (a).
#(1.3) Get the one-step estimator β̂ 1-step from (b).
#(1.4) Apply a standard implementation of logistic regression and obtain a β̂ GLM .
#(2) Make an illustration of the distributions of the three estimates in (c) under each sample size
#by e.g., boxplots similar to the figure below.

function simStudy(n)
    nums = 1:1000
    initVec    = Array{Float64}(undef, 1000, 2)
    oneStepVec = Array{Float64}(undef, 1000, 2)
    glmVec     = Array{Float64}(undef, 1000, 2)
    for i in eachindex(nums)
        R, Y, = simData(n, 0, 0);
        initVec[i,:]    = worseEst(R,Y, [0.0,0.0])
        oneStepVec[i,:] =  oneStep(R,Y, initVec[i, :])
        glmVec[i, :]     = glmEst(R,Y)
    end
    return (that = initVec, oneStepVec = oneStepVec, glmVec = glmVec)
end

#@benchmark simStudy(250)

#########################################################################
# MAKE PLOTS
#########################################################################

# Make plots with the following arrays of estimates
# a = worseEstimator (1 x n)
# b = Efficient weight matrix estimator: one-step (1 x n)
# c = glm fits (1 x n) 
function makeSimStudyPlot(a, b, c, n)
    p = violin(["A = (1, min(0.1, R))"], a)
    violin!(["Efficient A"], b)
    violin!(["glm"], c)
    boxplot!(["A = (1, min(0.1, R))"], a, fillalpha = 0.01, color = :black)
    boxplot!(["Efficient A"], b, fillalpha = 0.01, color = :black)
    boxplot!(["glm"], c, fillalpha = 0.01, color = :black)
    plot!(legend = false)
    plot!([0], seriestype="hline", line=:dash)
    ylabel!("\$\\hat{\\beta}\$")
    title!("n = " * string(n))
    ylims!((-3, 3))

    return p
end


#########################################################################
# SANDWICH ESTIMATOR
#########################################################################

# sandwich estimator of the vcov matrix
function sandwich(R,Y, θhat)
    n = length(Y)
    getAs(R) = [1 R]
    A = hcat((getAs.(R))'...)
    W = [expit2(θhat[1] + θhat[2] * Ri) for Ri in R]
    V = [expit(θhat[1] + θhat[2] * Ri) for Ri in R]
    outerProd = [A[:, i] * A[:, i]' for i in 1:n]
    EAD = 1/n * sum([W[i] .* outerProd[i] for i in 1:n])
    EAVA = 1/n * sum([(Y[i]-V[i])^2 .* outerProd[i] for i in 1:n])
    invEAD = inv(EAD)
    1/n .* invEAD * EAVA * invEAD'
end


# get distribution of difference between vcov(glm) and vcov(sandwich)
function checkSim(times, n)
    covaaVec = zeros(times)
    covabVec = zeros(times)
    covbbVec = zeros(times)

    for i in 1:times
        R, Y, = simData(n, 0, 0);
        θhat = worseEst(R,Y, [0.0, 0.0])
        s1 = oneStep(R,Y, θhat)
        fit  = glm(@formula(Y ~ R), (;R, Y), Binomial(), LogitLink())
        # extract var(α), cov(α, β), var(β) by flattening
        covaaVec[i],covabVec[i],_,covbbVec[i] = vcat((vcov(fit) - sandwich(R, Y, s1))...)
    end

    (aa = covaaVec, ab = covabVec, bb = covbbVec)
end

#########################################################################
# Make simulations and return plots.
#########################################################################

#
n = 250
@time a,b,c = simStudy(n)
# get estimates of β in 2. column from the three methods
a, b, c = a[:, 2], b[:, 2], c[:, 2]
p250 = makeSimStudyPlot(a, b, c, n)

n = 500
@time a,b,c = simStudy(n)
# get estimates of β in 2. column from the three methods
a, b, c = a[:, 2], b[:, 2], c[:, 2]
p500 = makeSimStudyPlot(a, b, c, n)
ylabel!("")

n = 1000
@time a,b,c = simStudy(n)
# get estimates of β in 2. column from the three methods
a, b, c = a[:, 2], b[:, 2], c[:, 2]
p1000 = makeSimStudyPlot(a, b, c, n)
ylabel!("")

l = @layout[a b c]
p = Plots.plot(p250, p500, p1000, layout = l, size = (1200, 400), 
            left_margin = 22px, 
            bottom_margin = 10px,
            top_margin = 10px)



@time aa, ab, bb = checkSim(100, 5000)
pdiff = violin(["var(α-glm) - var(α-sandwich)"], aa)
violin!(["var(β-glm) - var(β-sandwich)"], bb)
violin!(["cov(α-glm, β-glm) - cov(α-sandwich, β-sandwich)"], ab)
boxplot!(["var(α-glm) - var(α-sandwich)"], aa, fillalpha = 0.01, color = :black)
boxplot!(["var(β-glm) - var(β-sandwich)"], ab, fillalpha = 0.01, color = :black)
boxplot!(["cov(α-glm, β-glm) - cov(α-sandwich, β-sandwich)"], bb, fillalpha = 0.01, color = :black)
plot!([0], seriestype="hline", line=:dash)
Plots.plot!(size = (1200, 600), legend = false)
title!("Difference in estimates: n = 5000, repeat = 100" )

### UNCOMMENT TO SAVE PLOTS TO DISK IN THE PDF ###
#root = dirname(dirname(@__FILE__))
#Plots.savefig(p, root * "/exercise2/simStudyPlot.pdf" )
#Plots.savefig(pdiff, root * "/exercise2/diffPlot.pdf" )


