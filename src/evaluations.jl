function empirical_optimum_instance(probleminstance::DecisionProblem, kbound, xs, vs, T)
    (X, V, C, H, D, ms, ps, model_factory) = unpack(probleminstance)
    inds_p = [0; cumsum(probleminstance.ps)]
    inds_m = [0; cumsum(probleminstance.ms)]
    model = Model(model_factory)
    set_silent(model)
    @variable(model, norm_z)
    @variable(model, 0 ≤ t ≤ kbound)
    variables = Dict{Symbol,Array{VariableRef,2}}(
       Symbol("K$i") => @variable(model, [1:ms[i], 1:ps[i]], base_name = "K$i")
       for i in 1:Nplayers(probleminstance)
    )
    ys = C*xs[:, 1:T] + vs[:, 1:T]
    us = Matrix{AffExpr}(undef, udim(probleminstance), T)
    for i = 1:Nplayers(probleminstance)
        yis = ys[(inds_p[i] + 1) : inds_p[i + 1], :]
        us[(inds_m[i] + 1) : inds_m[i + 1], :] = variables[Symbol("K$i")] * yis
        # Bound the spectral norm of Ki by kbound
        @constraint(model, [kbound; vec(variables[Symbol("K$i")])] ∈ NormSpectralCone(ms[i], ps[i]))
    end
    zs = H * xs[:, 1:T] + D * us
    @constraint(model, [norm_z;vec(zs)] ∈ SecondOrderCone())
    @objective(model, Min, norm_z)
    optimize!(model)
    losses = sum(value.(zs).^2, dims = 1)
    Ks = Dict{Symbol,Matrix}(
       Symbol("K$i") => value.(variables[Symbol("K$i")])
       for i in 1:Nplayers(probleminstance)
    )
    return (Ks, losses[:])
end

function empirical_optimum(probleminstance::DecisionProblem, kbound, xs, vs)
    T = size(xs)[2]
    cumlosses = zeros(T)
    for t = 1:(T-1)
        _, lossest = empirical_optimum_instance(probleminstance, kbound, xs, vs, t)
        cumlosses[t] = sum(lossest)
    end
    Ks, lossesT = empirical_optimum_instance(probleminstance, kbound, xs, vs, T)
    cumlosses[T] = sum(lossesT)
    return (Ks, cumlosses)
end

function pseudo_optimum(probleminstance::DecisionProblem, lossfn, T)
    Ks = optimaltd(probleminstance)
    return (Ks, [lossfn(Ks, t) for t = 1:T])
end

function simulate_once(probleminstance::DecisionProblem, λ, T, kbound)
    println("Setting up a learning team decision problem.")
    println("Number of Players: $(Nplayers(probleminstance))")
    println("Stepsize parameter λ: $λ")
    println("Time Horizon: $T")
    println("Spectral-norm bound on gain matrices: $kbound")
    println("Strong convexity lower bound: $(strongconvexity(probleminstance))")

    println("Sampling nature...")
    stepsizes = t -> 1 / λ / t
    explorationfn = get_explorationfn(probleminstance)
    (lossfn, gradientfn, xs, vs) = sample_nature(probleminstance, T);
    K0s = Dict{Symbol,Matrix}(Symbol("K$i") => randn(probleminstance.ms[i],probleminstance.ps[i])
        for i ∈ 1:LearningTeamDecisions.Nplayers(probleminstance))

    println("Learning with gradients...")
    (Ks1, losses1) = learning_with_gradients(K0s, kbound, lossfn, gradientfn, stepsizes, T)
    println("Total loss: $(sum(losses1))")
    println("Learning with bandit feedback")
    (Ks2, losses2) = learning_bandit(K0s, kbound, lossfn, stepsizes, explorationfn, T)
    println("Total loss: $(sum(losses2))")
    println("Solving for realization-dependent optimum:")
    (Ks3, cumlosses3) = empirical_optimum(probleminstance, kbound, xs, vs);
    println("Total loss: $(cumlosses3[end])")
    println("Computing pseudo optimum")
    (Ks4, losses4) = pseudo_optimum(probleminstance, lossfn, T);
    println("Total loss: $(sum(losses4))")
    println("Plotting...")
    p = plot(1:T,cumsum(losses1), label = "w. Gradient", linewidth = :4)
    plot!(1:T,cumsum(losses2), label = "bandit feedback", linewidth = :4)
    plot!(1:T,cumlosses3, label = "Empirical optimal", linewidth = :4)
    plot!(1:T,cumsum(losses4), label = "Pseudo optimal", linewidth = :4)
    plot!(p, yaxis = :log, xaxis = :log, legend = :bottomright, xlabel = "t", ylabel = "sum(|z(τ)|²)")
    plot!(p, title="loglog plot")
    return p
end