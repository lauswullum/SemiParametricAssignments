# SemiParametricAssignments

Code written for three assignments for the course semi-parametric inference

## Running experiments and getting output
To run the experiments and applied study you run the following in your terminal

```Julia
julia appliedTest.jl
```
This produces a latex file `latexApplied.tex` containing a table with results
in latex format as well as printouts of the estimates and confidence intervals. 

```Julia
julia simstudy3.jl
```
This produces a latex file `latexSimstudy.tex` containing a table with results
in latex format as well as printouts of the study. 


## Required packages in Julia
* Distributions  
* DataFrames  
* DataFramesMeta  
* GLM  
* Statistics  
* Plots  
* StatsPlots  
* Plots.PlotMeasures  
* QuadGK  
* NLsolve  
* LaTeXStrings  
* ForwardDiff  
* PrettyTables  


## Required packages in R
* gt  
* magrittr  
* tibble  
* tidyr  
* dplyr  
* reshape2  

## Remark
These scripts run on Julia version 1.7.3.  
No further work done to ensure reproducibility.  
