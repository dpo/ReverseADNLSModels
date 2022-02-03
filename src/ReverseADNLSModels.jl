module ReverseADNLSModels
using LinearAlgebra
using ReverseDiff
using NLPModels

export ReverseADNLSModel

"""
    model = ReverseADNLSModel(r!; name = "reverse AD NLS model")

A simple subtype of `AbstractNLSModel` to represent a nonlinear least-squares problem
with a smooth residual Jacobian-vector products computed via reverse-mode AD.

## Arguments

* `r! :: R <: Function`: a function such that `r!(y, x)` stores the residual at `x` in `y`.
"""
mutable struct ReverseADNLSModel{T, S, R} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  _tmp_input::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _tmp_output::S
  _tmp_output_rd::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}

  function ReverseADNLSModel{T, S, R}(
    r::R,
    nequ::Int,
    x::S;
    name::AbstractString = "reverse AD NLS model",
  ) where {T <: Real, S, R <: Function}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x, name = name)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    tmp_input = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
    tmp_output = S(undef, nequ)
    tmp_output_rd = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nequ)
    return new{T, S, R}(meta, nls_meta, NLSCounters(), r, tmp_input, tmp_output, tmp_output_rd)
  end
end

# convenience constructor
ReverseADNLSModel(r, nequ::Int, x::S; kwargs...) where {S} =
  ReverseADNLSModel{eltype(S), S, typeof(r)}(
    r,
    nequ,
    x;
    kwargs...,
  )

function NLPModels.residual!(nls::ReverseADNLSModel, x::AbstractVector, Fx::AbstractVector)
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  nls.resid!(Fx, x)
  Fx
end

function NLPModels.jprod_residual!(
  nls::ReverseADNLSModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Jv::AbstractVector{T},
) where T
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  # J(x) * v is the derivative at t = 0 of t ↦ r(x + tv)
  ϕ!(out, t) = begin
    # here t is a vector of ReverseDiff.TrackedReal
    nls._tmp_input .= x .+ t[1] .* v
    nls.resid!(out, nls._tmp_input)
    out
  end
  ReverseDiff.jacobian!(Jv, ϕ!, nls._tmp_output, [zero(T)])
  Jv
end

function NLPModels.jtprod_residual!(
  nls::ReverseADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  # J(x)' * v is the gradient of x ↦ r(x)' * v.
  ϕ(u) = begin
    # here u is a vector of ReverseDiff.TrackedReal
    nls.resid!(nls._tmp_output_rd, u)
    dot(nls._tmp_output_rd, v)
  end
  ReverseDiff.gradient!(Jtv, ϕ, x)
  Jtv
end

end # module