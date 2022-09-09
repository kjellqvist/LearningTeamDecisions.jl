# Define problem parameters
using Mosek, MosekTools
using LearningTeamDecisions
using JLD
using Dates
using ProgressMeter

run_time=now()
run_dir ="run_$run_time"
mkdir(run_dir)

include("preamble.jl")

X = fill(1, 1,1)
V = [1 0;0 1]
C = fill(1, 2, 1)
H = [1 0 0]'
D = [1 1; 1 0; 0 1]

ms = [1; 1]
ps = [1; 1]
model_factory = Mosek.Optimizer
probleminstance = DecisionProblem{Float64, Int64}(X, V, C, H, D, ms, ps, model_factory)
#
T = 1000
λ = 1
kbound = 2

K0s = Dict{Symbol,Matrix}(:K1 => fill(randn(), 1,1), :K2 => fill(randn(),1,1))
##
function one_run(probleminstance::DecisionProblem, λ, T, kbound)
    stepsizes = t -> 1 / λ / t
    explorationfn = get_explorationfn(probleminstance)
    (lossfn, gradientfn, xs, vs) = sample_nature(probleminstance, T);
    K0s = Dict{Symbol,Matrix}(Symbol("K$i") => randn(probleminstance.ms[i],probleminstance.ps[i])
        for i ∈ 1:LearningTeamDecisions.Nplayers(probleminstance))

    (_, losses1) = learning_with_gradients(K0s, kbound, lossfn, gradientfn, stepsizes, T)
    (_, losses2) = learning_bandit(K0s, kbound, lossfn, stepsizes, explorationfn, T)
    (_, cumlosses3) = empirical_optimum(probleminstance, kbound, xs, vs);
    (_, losses4) = pseudo_optimum(probleminstance, lossfn, T);
        return (losses1, losses2, cumlosses3, losses4)
end

##
Nexperiments = 1280
T = 1000
println("Setting up a learning team decision problem.")
println("Number of Players: $(LearningTeamDecisions.Nplayers(probleminstance))")
println("Stepsize parameter λ: $λ")
println("Time Horizon: $T")
println("Spectral-norm bound on gain matrices: $kbound")
println("Strong convexity lower bound: $(LearningTeamDecisions.strongconvexity(probleminstance))")
println("Running $Nexperiments experiments")

losses_gradient = zeros(Nexperiments, T)
losses_bandit = zeros(Nexperiments, T)
losses_empirical = zeros(Nexperiments, T)
losses_pseudo = zeros(Nexperiments, T)

p = Progress(Nexperiments);
update!(p,0)
jj = Threads.Atomic{Int}(0)
l = Threads.SpinLock()
Threads.@threads for i = 1:Nexperiments
    (losses_gradient[i, :], losses_bandit[i, :], losses_empirical[i, :], losses_pseudo[i, :]) = one_run(probleminstance, λ, T, kbound)
    Threads.atomic_add!(jj, 1)
    Threads.lock(l)
    update!(p, jj[])
    Threads.unlock(l)  
end

save("$run_dir/data.jld", 
    "losses_gradient", losses_gradient, 
    "losses_bandit", losses_bandit, 
    "losses_empirical", losses_empirical, 
    "losses_pseudo", losses_pseudo)
println("Saved to $run_dir/data.jld")
close(logger)