using Distributions
using DataFrames
using DataFramesMeta
using GLM
using Statistics
using Plots
using Plots.PlotMeasures
using QuadGK

############################################
# Functions to conduct simulation study
############################################

# Global settings for the given data generating process
πR = 0.5
σX = 2
σYgivenRX = 1

function expit(x)
    1 / (1 + exp(-x))
end

function expitAlt(x)
    1 / (1 + exp(x))
end

# Simulation study for linear regression
function generateSimData(n, β, γ, modeltype=0)
    r = rand(Bernoulli(πR), n)
    x = rand(Normal(0, σX), n)
    condMean = β * r + γ * x
    if modeltype == 0
        y = @. rand(Normal(condMean, σYgivenRX))
    else
        pVec = expit.(condMean)
        y = @. rand(Bernoulli(pVec))
    end
    return DataFrame(x=x, r=r, y=y)
end

# Fit data using GLM.jl
function fitAndGetEstimates(simData, modeltype=0)
    if modeltype == 0
        # Fit models with and without covariate
        fitWith = lm(@formula(y ~ r + x), simData)
        fitWithOut = lm(@formula(y ~ r), simData)
    else
        fitWith = glm(@formula(y ~ r + x), simData, Binomial(), LogitLink())
        fitWithOut = glm(@formula(y ~ r), simData, Binomial(), LogitLink())
    end

    # Extract estimates and their sd's
    betaWith = coef(fitWith)[2]
    betaWithStd = stderror(fitWith)[2]
    betaWithOut = coef(fitWithOut)[2]
    betaWithOutStd = stderror(fitWithOut)[2]

    return (bw=betaWith,
        bwstd=betaWithStd,
        bwo=betaWithOut,
        bwostd=betaWithOutStd)
end

# Get mean bias and mean relative efficiency over 1000 simulated dataset
# of size param:n 
function simStudy(n, β, γ, modeltype)
    numrep = 1000
    βwSimList = zeros(numrep)
    βwSimStdList = zeros(numrep)
    βwoSimList = zeros(numrep)
    βwoSimStdList = zeros(numrep)
    # Conduct simulation study
    for i in 1:numrep
        simData = generateSimData(n, β, γ, modeltype)
        (βwSimList[i], βwSimStdList[i], βwoSimList[i], βwoSimStdList[i]) =
            fitAndGetEstimates(simData, modeltype)
    end

    biasw  = mean(βwSimList .- β)
    biaswo = mean(βwoSimList .- β)
    relEff = mean(@. (βwoSimStdList * βwoSimStdList) / (βwSimStdList * βwSimStdList))
    βw     = mean(βwSimList)
    βwo    = mean(βwoSimList)
    
    return (biasw, biaswo, relEff, βw, βwo)
end

# Conduct the simulation study for the different tuples of parameter values
# specified below
function combinationsSimStudy(n, βlist, γlist, modeltype)
    nβ = length(βlist)
    nγ = length(γlist)

    # Matrices to store results
    matCondEffect = zeros((nβ, nγ))
    matMargEffect = zeros((nβ, nγ))
    matRelEff = zeros((nβ, nγ))
    matβw = zeros((nβ, nγ))
    matβwo = zeros((nβ, nγ))

    # Loop over different combinations of parameter values. 
    for i in eachindex(βlist)
        for j in eachindex(γlist)
            biasw, biaswo, relEff, βw, βwo = simStudy(n, βlist[i], γlist[j], modeltype)
            matCondEffect[i, j] = biasw
            matMargEffect[i, j] = biaswo
            matRelEff[i, j] = relEff
            matβw[i, j] = βw
            matβwo[i, j] = βwo
        end
    end
    return (mc=matCondEffect, mm=matMargEffect, mrel=matRelEff, βw=matβw, βwo=matβwo)
end


βlist = [-log(2), 0, log(2)]
γlist = [-log(4), -log(2), 0, log(2), log(4)]
γlistCat = ["-log(4)", "-log(2)", "0", "log(2)", "log(4)"]

###################################
# LINEAR REGRESSION STUDY
###################################

@time mclin, mmlin, mrelin, _, _ = combinationsSimStudy(500, βlist, γlist, 0)


# Conditional effect, bias
p11 = scatter(γlistCat, mclin[1, :], title="Linear: β = -log(2)")
plot!(p11, [0], seriestype="hline", line=:dash)
ylabel!(p11, "Bias, Conditional effect")
p12 = scatter(γlistCat, mclin[2, :], title="Linear: β = 0")
plot!(p12, [0], seriestype="hline", line=:dash)
p13 = scatter(γlistCat, mclin[3, :], title="Linear: β = log(2)")
plot!(p13, [0], seriestype="hline", line=:dash)
pmc = plot(p11, p12, p13,layout=(1, 3),size=(1500, 300),legend=false)
ylims!(pmc, -1, 1)

# Marginal effect, bias
p11 = scatter(γlistCat, mmlin[1, :], title="Linear: β = -log(2)")
plot!(p11, [0], seriestype="hline", line=:dash)
ylabel!(p11, "Bias, Marginal effect")
p12 = scatter(γlistCat, mmlin[2, :], title="Linear: β = 0")
plot!(p12, [0], seriestype="hline", line=:dash)
p13 = scatter(γlistCat, mmlin[3, :], title="Linear: β = log(2)")
plot!(p13, [0], seriestype="hline", line=:dash)
pmm = plot(p11, p12, p13, layout=(1, 3), size=(1500, 300),legend=false)
ylims!(pmm, -1, 1)


# Relative Efficiency plots
p11 = scatter(γlistCat, mrelin[1, :], title="Linear: β = -log(2)")
plot!(p11, [1], seriestype="hline", line=:dash)
ylabel!(p11, "Relative efficiency")
xlabel!(p11, "γ")
p12 = scatter(γlistCat, mrelin[2, :], title="Linear: β = 0")
plot!(p12, [1], seriestype="hline", line=:dash)
xlabel!(p12, "γ")
p13 = scatter(γlistCat, mrelin[3, :], title="Linear: β = log(2)")
plot!(p13, [1], seriestype="hline", line=:dash)
xlabel!(p13, "γ")
pmrel = plot(p11, p12, p13,
    layout=(1, 3),
    size=(1500, 300),
    legend=false)

pf = plot!(pmc, pmm, pmrel, layout=(3, 1), size=(1100, 750), plot_title = "Linear Model",
            bottom_margin=50px, left_margin=60px, right_margin=30px, top_margin=15px)

### UNCOMMENT TO WRITE PLOTS TO DISK ###
#root = dirname(dirname(@__FILE__))
#savefig(pf, root * "/linearPlot1.pdf" )


###################################
# LOGISTIC REGRESSION STUDY
###################################

@time mc1, mm1, mrel, βwList, βwoList  = combinationsSimStudy(500, βlist, γlist, 1)

# Conditional effect, bias
p11 = scatter(γlistCat, mc1[1, :], title="Linear: β = -log(2)")
plot!(p11, [0], seriestype="hline", line=:dash)
ylabel!(p11, "Bias, Conditional effect")
p12 = scatter(γlistCat, mc1[2, :], title="Linear: β = 0")
plot!(p12, [0], seriestype="hline", line=:dash)
p13 = scatter(γlistCat, mc1[3, :], title="Linear: β = log(2)")
plot!(p13, [0], seriestype="hline", line=:dash)
pmcL = plot(p11, p12, p13,layout=(1, 3),size=(1500, 300),legend=false)
ylims!(-1, 1)


# Marginal effect, bias
p11 = scatter(γlistCat, mm1[1, :], title="Linear: β = -log(2)")
plot!(p11, [0], seriestype="hline", line=:dash)
ylabel!(p11, "Bias, Marginal effect")
xlabel!(p11, "γ")
p12 = scatter(γlistCat, mm1[2, :], title="Linear: β = 0")
plot!(p12, [0], seriestype="hline", line=:dash)
xlabel!(p12, "γ")
p13 = scatter(γlistCat, mm1[3, :], title="Linear: β = log(2)")
plot!(p13, [0], seriestype="hline", line=:dash)
xlabel!(p13, "γ")
pmmL = plot(p11, p12, p13, layout=(1, 3), size=(1500, 300),legend=false)
ylims!(-1, 1)


pfL = plot(pmcL, pmmL, layout=(2, 1), size=(1100, 600), plot_title = "Logistic model",
            bottom_margin=50px, left_margin=60px, right_margin=30px, top_margin=15px)

### UNCOMMENT TO WRITE PLOTS TO DISK ###
#root = dirname(dirname(@__FILE__))
#savefig(pfL, root * "/logisticPlot1.pdf" )

# Relative eff: It is higher for the adjusted than for the non-adjusted

##############################################
# Marginal log odds via numerical integration
##############################################

# Compute the theoretical marginal Log OR using numerical integration using QuadGK.jl
function logOREZ(β)
    function logORZ(γ)
        # Get normal pdf
        normalpdf(x) = pdf(Normal(0, σX), x)
        #
        pY1R1(x) = normalpdf(x) * expit(β + γ*x)
        #pY0R1(x) = normalPdf(x) * expitAlt(β + γ*x)
        # and
        pY1R0(x) = normalpdf(x) * expit(γ*x)
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

macro Name(arg)
    string(arg)
end

# Theoretical logOR plotter
function genPlotForLogOR(β, βwList, βwoList)
    fw = logOREZ(β)
    grid = range(-log(8), log(8), length = 200)
    titleList = [4, 2, 0, 2, 4]
    γList = [-log(4), -log(2), 0, log(2), log(4)]
    γticks = [-log(8), -log(4), -log(2), 0, log(2), log(4), log(8)]
    γticksStr = ["-log(8)", "-log(4)", "-log(2)", "0", "log(2)", "log(4)", "log(8)"]
    plottet = plot(grid, fw,label = "Theoretical logOR")
    xlabel!(plottet, "γ")
    ylabel!(plottet, "log OR")

    # Should we add estimates of β
    whenToAddPoints = [-log(2), 0, log(2)]
    if β in whenToAddPoints 
        ind = findall(x -> x == β, whenToAddPoints) |> first
        print(ind)
        scatter!(γList, βwoList[ind, :], label = "No adjustment")
        scatter!(γList, βwList[ind, :], label = "Adjusted")
    end
    # Should the log be negative of positive in the title
    ind2 = findall(x -> x == β, γList) |> first
    if ind2 > 2
        titleStr = @Name(β) * " = log(" * string(titleList[ind2]) * ")" 
    else
        titleStr = @Name(β) * " = -log(" * string(titleList[ind2]) * ")" 
    end
    plot!(plottet, xticks = (γticks, γticksStr), title = titleStr)
    return plottet
end


pmlog4 = genPlotForLogOR(-log(4), βwList, βwoList)
plot!(pmlog4, legend = false)
pmlog2 = genPlotForLogOR(-log(2), βwList, βwoList)
plot!(pmlog2, legend = false)
p0 = genPlotForLogOR(0, βwList, βwoList)
ylims!(p0, (-1,1))
plot!(p0, title = "β = 0")
plog2 = genPlotForLogOR(log(2), βwList, βwoList)
plot!(plog2, legend = false)
plog4 = genPlotForLogOR(log(4), βwList, βwoList)
plot!(plog4, legend = false)

l = @layout [a b ; c b ; d _]
p3 = plot(pmlog4, plog4, pmlog2, plog2, p0, layout = l, size = (1000, 900), bottom_margin=10px, left_margin=10px, right_margin=10px, top_margin=5px)
#p3 = plot(p1, p0, layout = (2,1),size = (900, 900), bottom_margin=10px, left_margin=10px, right_margin=10px, top_margin=5px)

### UNCOMMENT TO WRITE PLOTS TO DISK ###
#root = dirname(dirname(@__FILE__))
#savefig(p3, root * "/logOR1.pdf" )


##############################################
# Asymptotic bias plot
##############################################


hpp(x) = (exp(-x)*(exp(-x)- 1)) / (1 + exp(-x))^3

hp(x) = (exp(-x)) / (1 + exp(-x))^2

# A theoretical bias found by using several taylor expansions. 
# This theoretical asymptotic bias is plotted against the simulated bias.
function asympBiasFactory(β)
    function plotter(γ)
        0.5 * γ * γ * σX * σX * ( ( hpp(β) ) / hp(β) )
    end
end


p11 = scatter(γlist, mm1[1, :], title="Linear: β = -log(2)", legend = false)
plot!(p11, [0], seriestype="hline", line=:dash)
ylabel!(p11, "Bias, Marginal effect")
xlabel!(p11, "γ")
plotBias = asympBiasFactory(-log(2))
plot!(plotBias)
p12 = scatter(γlist, mm1[2, :], title="Linear: β = 0")
plot!(p12, [0], seriestype="hline", line=:dash)
xlabel!(p12, "γ")
plotBias = asympBiasFactory(0)
plot!(plotBias, legend = false)
p13 = scatter(γlist, mm1[3, :], title="Linear: β = log(2)", label = "No Adjustment")
plot!(p13, [0], seriestype="hline", line=:dash, label = "y = 0")
xlabel!(p13, "γ")
plotBias = asympBiasFactory(log(2))
plot!(plotBias, label = "Asymptotic bias")
γticks = [-log(4), -log(2), 0, log(2), log(4)]
γticksStr = ["-log(4)", "-log(2)", "0", "log(2)", "log(4)"]
pmmLAs = plot(p11, p12, p13, layout=(1, 3), size=(1000, 300), bottom_margin=20px, left_margin=19px, right_margin=10px, top_margin=5px)
plot!(pmmLAs, xticks = (γticks, γticksStr))
ylims!(pmmLAs, -1, 1)


### UNCOMMENT TO WRITE PLOTS TO DISK ###
#root = dirname(dirname(@__FILE__))
#savefig(pmmLAs, root * "/asympBiasCond.pdf" )

