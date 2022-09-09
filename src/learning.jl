function sample_nature(probleminstance::DecisionProblem{R, I}, T::Integer) where R<:Real where I<:Integer
    xs = sqrt(probleminstance.X) * randn(xdim(probleminstance), T)
    vs = sqrt(probleminstance.V) * randn(ydim(probleminstance), T)
    inds_p = [0; cumsum(probleminstance.ps)]
    inds_m = [0; cumsum(probleminstance.ms)]
    
    function _get_z_and_y(Ks::Dict{Symbol,M}, t) where M <: AbstractMatrix
        x = xs[:, t]
        v = vs[:, t]
        u = zeros(udim(probleminstance))
        y = zeros(ydim(probleminstance))
        
        for i = 1:Nplayers(probleminstance)
            yi = probleminstance.C[(inds_p[i] + 1) : inds_p[i + 1], :] * x + v[(inds_p[i] + 1) : inds_p[i + 1]]
            u[(inds_m[i] + 1) : inds_m[i + 1]] = Ks[Symbol("K$i")] * yi
            y[(inds_p[i] + 1) : inds_p[i + 1]] = yi
        end
        z = probleminstance.H*x + probleminstance.D*u
        return (z, y)
    end
    
    function loss(Ks::Dict{Symbol,M}, t) where M <: AbstractMatrix
        z = _get_z_and_y(Ks, t)[1]
        return z'*z
    end
    
    function gradients(Ks::Dict{Symbol,M}, t) where M <: AbstractMatrix
        (z, y) = _get_z_and_y(Ks, t)
        grads = Dict{Symbol,Matrix}(
           Symbol("G$i") => 2*probleminstance.D[:, (inds_m[i] + 1) : inds_m[i + 1]]' * z * y[(inds_p[i] + 1) : inds_p[i + 1]]'
           for i in 1:Nplayers(probleminstance)
            )
        return grads
    end
    return (loss, gradients, xs, vs)
end

function gradientbound(pi::DecisionProblem, kbound::Real)
    bg = 4 * opnorm(pi.D)^2 * (opnorm(pi.H) + kbound * opnorm(pi.D) * (opnorm(pi.C) + 1))^2 *
        (opnorm(pi.C) + 1)^2 * (fourthmoment(pi.X) + 2 * tr(pi.X) * tr(pi.V) + fourthmoment(pi.V)) |> sqrt
    return bg
end

function regretbound_gradient(pi::DecisionProblem, λ::Real, kbound::Real)
    @assert 0 < λ < strongconvexity(pi) "λ was chosen larger than the strong convexity parameter, or negative"
    return T -> gradientbound(pi, kbound)^2 * (1 + log(T))
end

function regretbound_bandit(pi::DecisionProblem, λ::Real, kbound::Real)
    M1 = opnorm(pi.D)^2 *(opnorm(pi.C)^2 * tr(pi.X) + tr(pi.V))
    M2 = (opnorm(pi.H) + opnorm(pi.D) * (kbound + 1) * (opnorm(pi.C) + 1))^4 * 
        (fourthmoment(pi.X) + 2 * tr(pi.X) * tr(pi.V) + fourthmoment(pi.V))
    return T-> 2*(M1 + M2/λ) * norm(pi.ms .* pi.ps)*sqrt(T)
end

function learning_with_gradients(K0s, kbound, lossfn, gradientfn, stepsizes, timehorizon)
    Ks = copy(K0s)
    losses = zeros(timehorizon)
    for t = 1:timehorizon
        losses[t] = lossfn(Ks, t)
        gradients = gradientfn(Ks, t)
        for i = 1:length(K0s)
            gradient_step = Ks[Symbol("K$i")] - stepsizes(t) * gradients[Symbol("G$i")]
            Ks[Symbol("K$i")] = gradient_step * min(opnorm(gradient_step), kbound) / opnorm(gradient_step) 
        end
    end
    return (Ks, losses)
end

function learning_bandit(K0s, kbound, lossfn, stepsizes, explorationfn, timehorizon)
    Ks = copy(K0s)
    perturbedKs = copy(K0s)
    Rs = copy(K0s)
    losses = zeros(timehorizon)
    for t = 1:timehorizon
        for i = 1:length(K0s)
            Rs[Symbol("K$i")] = rand((-1, 1), size(Rs[Symbol("K$i")]))
            perturbedKs[Symbol("K$i")] = Ks[Symbol("K$i")] + explorationfn(i, t) * Rs[Symbol("K$i")]
        end
        losses[t] = lossfn(perturbedKs, t)
        for i = 1:length(K0s)
            gradient_estimate = losses[t] *  Rs[Symbol("K$i")] / explorationfn(i, t)
            gradient_step = Ks[Symbol("K$i")] - stepsizes(t) * gradient_estimate
            Ks[Symbol("K$i")] = gradient_step * min(opnorm(gradient_step), kbound) / opnorm(gradient_step)
        end
    end
    return (Ks, losses)
end

function get_explorationfn(probleminstance::DecisionProblem)
    (ms, ps) = (probleminstance.ms, probleminstance.ps)
    dimensional_scaling = sum(ms.^2 .* ps.^2)^(-1/4)
    time_scaling = t -> t^(-1/4)
    
    function explorationfn(i, t)
        return time_scaling(t) * dimensional_scaling / sqrt(ms[i] * ps[i])        
    end
    return explorationfn
end

