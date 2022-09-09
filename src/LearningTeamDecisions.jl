module LearningTeamDecisions
    using LinearAlgebra
    using JuMP
    using MathOptInterface:NormSpectralCone
    using Plots

    export DecisionProblem
    export sample_nature, gradientbound, regretbound_bandit, regretbound_gradient
    export learning_with_gradients, learning_bandit, get_explorationfn
    export empirical_optimum, pseudo_optimum, simulate_once

    include("decisionproblems.jl")
    include("evaluations.jl")
    include("learning.jl")
end
