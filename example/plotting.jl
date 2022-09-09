using JLD
using Statistics
using Plots
using DataFrames
using CSV
run_dir = "run_2022-09-09T12:55:47.755"
d = load(run_dir * "/data.jld")
mean(d["losses_gradient"], dims=1)
std(d["losses_gradient"], dims=1)

regret_gradient = cumsum(d["losses_gradient"], dims = 2) - d["losses_empirical"]
regret_bandit = cumsum(d["losses_bandit"], dims = 2) - d["losses_empirical"]
regret_pseudo = cumsum(d["losses_pseudo"], dims = 2) - d["losses_empirical"]


min_gradient = minimum(regret_gradient, dims = 1)
avg_gradient = mean(regret_gradient, dims = 1)
max_gradient = maximum(regret_gradient, dims = 1)
std_gradient = std(regret_gradient, dims = 1)

min_bandit = minimum(regret_bandit, dims = 1)
avg_bandit = mean(regret_bandit, dims = 1)
max_bandit = maximum(regret_bandit, dims = 1)
std_bandit = std(regret_bandit, dims = 1)

min_pseudo = minimum(regret_pseudo, dims = 1)
avg_pseudo = mean(regret_pseudo, dims = 1)
max_pseudo = maximum(regret_pseudo, dims = 1)
std_pseudo = std(regret_pseudo, dims = 1)

T = length(min_gradient)

plot(1:T, avg_gradient',ribbon = std_gradient', fillalpha = 0.35, c = 1, lab="Mean, Gradient feedback",  linewidth = :4, linecolor = :blue,
xaxis = :lin, yaxis = :lin, ylims = (1, maximum(max_bandit)))
plot!(1:T, avg_bandit', ribbon = std_bandit', lab="Mean, Bandit", linewidth = :4, linecolor = :red)
plot(1:T, avg_pseudo', ribbon = std_pseudo', lab="Mean, Pseudo", linewidth = :4, linecolor = :green)
grad = regretbound_gradient(probleminstance, λ, kbound)
band = regretbound_bandit(probleminstance, λ, kbound)
##
df = DataFrame(
    t               = 1:T,
    min_gradient    = min_gradient[:],
    avg_gradient    = avg_gradient[:],
    max_gradient    = max_gradient[:],
    std_gradient    = std_gradient[:],
    min_bandit      = min_bandit[:],
    avg_bandit      = avg_bandit[:],
    max_bandit      = max_bandit[:],
    std_bandit      = std_bandit[:],
    min_pseudo      = min_pseudo[:],
    avg_pseudo      = avg_pseudo[:],
    max_pseudo      = max_pseudo[:],
    std_pseudo      = std_pseudo[:]
)

CSV.write(run_dir * "/stats.csv", df)