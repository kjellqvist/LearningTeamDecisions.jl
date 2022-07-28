struct DecisionProblem{R<:Real, I<:Integer}
    X::AbstractMatrix{R} 
    V::AbstractMatrix{R} 
    C::AbstractMatrix{R}
    H::AbstractMatrix{R} 
    D::AbstractMatrix{R} 
    ms::AbstractVector{I} 
    ps::AbstractVector{I}
    model_factory
end

function unpack(pi::DecisionProblem{R, I}) where R<:Real where I<:Integer
    return (pi.X, pi.V, pi.C, pi.H, pi.D, pi.ms, pi.ps, pi.model_factory)
end

function DecisionProblem(
        X::AbstractMatrix, 
        V::AbstractMatrix, 
        C::AbstractMatrix, 
        H::AbstractMatrix, 
        D::AbstractMatrix, 
        ms::AbstractVector, 
        ps::AbstractVector, 
        model_factory)
    return DecisionProblem(Matrix{Float64}.((X, V, C, H, D))..., Vector{Int64}.((ms, ps))..., model_factory)
end

function xdim(pi::DecisionProblem{R, I})::I where R<:Real where I<:Integer
    return I(size(pi.H)[2])
end
function udim(pi::DecisionProblem{R, I})::I where R<:Real where I<:Integer
    return sum(pi.ms)
end
function ydim(pi::DecisionProblem{R, I})::I where R<:Real where I<:Integer
    return sum(pi.ps)
end
function Nplayers(pi::DecisionProblem{R, I})::I where R<:Real where I<:Integer
    return I(length(pi.ms))
end

function optimaltd(probleminstance::DecisionProblem{R, I}) where R<:Real where I<:Integer
    (X, V, C, H, D, ms, ps, model_factory) = unpack(probleminstance)
    Q = [H D]' * [H D]
    model = Model(model_factory) # Create a JuMP model
    set_silent(model)
    # Check that each diagonal block of V is positive definite
    m = sum(ms)
    @assert Nplayers(probleminstance) == length(ps) "length(ms) does not equal length(ps)"
    inds_p = [0; cumsum(ps)]
    inds_m = [0; cumsum(ms)]
    for i = 1:length(ps)
        Vii = V[(inds_p[i] + 1) : inds_p[i + 1], (inds_p[i] + 1) : inds_p[i + 1]]
        @assert isposdef(Vii) "Vii is not positive definite for i = $i"
    end
    # Partition Q
    Quu = Q[end - m + 1:end, end - m + 1:end]
    Qxx = Q[1:end - m, 1:end - m]
    Qxu = Q[1:end - m, end - m + 1: end]
    Qux = Qxu'
    # Quu must be positive
    @assert isposdef(Quu) "Quu must be positive definite"
    L =  - Quu\Qux # Optimal policy in the centralized, full information case.
    variables = Dict{Symbol,Array{VariableRef,2}}(
       Symbol("K$i") => @variable(model, [1:ms[i], 1:ps[i]], base_name = "K$i")
       for i in 1:Nplayers(probleminstance)
    )
    # Construct a linear feasibility program to get optimal soluton
    for i = 1:Nplayers(probleminstance)
        s = 0.
        Ci = C[(inds_p[i] + 1) : inds_p[i + 1], :]
        Quxi = Qux[(inds_m[i] + 1) : inds_m[i + 1], :]
        for j = 1:Nplayers(probleminstance)
            Quuij= Quu[(inds_m[i] + 1) : inds_m[i + 1], (inds_m[j] + 1):inds_m[j + 1]]
            Vji = V[(inds_p[j] + 1) : inds_p[j + 1], (inds_p[i] + 1) : inds_p[i + 1]]
            Cj = C[(inds_p[j] + 1) : inds_p[j + 1], :]
            s = s .+ Quuij * variables[Symbol("K$i")] * (Cj * X * Ci' + Vji)
        end
        @constraint(model, s .== -Quxi * X * Ci')
    end
    optimize!(model)
    Ks = Dict{Symbol,Matrix}(
       Symbol("K$i") => value.(variables[Symbol("K$i")])
       for i in 1:Nplayers(probleminstance)
    )
    return Ks
end

function fourthmoment(X::AbstractMatrix{<:Real})
    lambdas = eigvals(X)
    return sum(kron(lambdas, lambdas)) + 2*sum(lambdas.^2)
end


function strongconvexity(pi::DecisionProblem)
    return 2 * minimum(svdvals(pi.D' * pi.D)) * (minimum(svdvals(pi.C * pi.X * pi.C')) + minimum(svdvals(pi.V)))
end