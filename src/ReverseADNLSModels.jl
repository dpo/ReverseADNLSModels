module ReverseADNLSModels
using LinearAlgebra
using ReverseDiff
using NLPModels

export ReverseADNLSModel

abstract type ADBackend end

struct ReverseDiffAD{T, F1, F2} <: ADBackend where {T, F1 <: Function, F2 <: Function}
  ϕ!::F1
  ψ::F2
  _tmp_input::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _tmp_output::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  z::Vector{T}
end

function ReverseDiffAD(r!::R, T::DataType, nvar::Int, nequ::Int) where R <: Function
  # define ReverseDiff AD backend
  # ... auxiliary function for J(x) * v
  # ... J(x) * v is the derivative at t = 0 of t ↦ r(x + tv)
  ϕ!(out, t, x, v, tmp_in) = begin
    # here t is a vector of ReverseDiff.TrackedReal
    tmp_in .= x .+ t[1] .* v
    r!(out, tmp_in)
    out
  end
  # ... auxiliary function for J(x)' * v
  # ... J(x)' * v is the gradient of x ↦ r(x)' * v.
  ψ(u, v, tmp_out) = begin
    # here u is a vector of ReverseDiff.TrackedReal
    r!(tmp_out, u)
    dot(tmp_out, v)
  end
  # ... temporary storage
  _tmp_input_rd = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
  _tmp_output_rd = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nequ)
  ReverseDiffAD{T, typeof(ϕ!), typeof(ψ)}(ϕ!, ψ, _tmp_input_rd, _tmp_output_rd, [zero(T)])
end

function jprod_residual!(Jv, rd::ReverseDiffAD{T, F1, F2}, x, v, tmp_out) where {T, F1 <: Function, F2 <: Function}
  ReverseDiff.jacobian!(Jv, (out, t) -> rd.ϕ!(out, t, x, v, rd._tmp_input), tmp_out, rd.z)
  Jv
end

function jtprod_residual!(Jtv, rd::ReverseDiffAD{T, F1, F2}, x, v) where {T, F1 <: Function, F2 <: Function}
  ReverseDiff.gradient!(Jtv, u -> rd.ψ(u, v, rd._tmp_output), x)
  Jtv
end

"""
    model = ReverseADNLSModel(r!; name = "reverse AD NLS model")

A simple subtype of `AbstractNLSModel` to represent a nonlinear least-squares problem
with a smooth residual Jacobian-vector products computed via reverse-mode AD.

## Arguments

* `r! :: R <: Function`: a function such that `r!(y, x)` stores the residual at `x` in `y`.
"""
mutable struct ReverseADNLSModel{T, S, R, AD} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  _tmp_output::S
  rd::AD

  function ReverseADNLSModel{T, S, R, AD}(
    r!::R,
    nequ::Int,
    x::S,
    rd::AD;
    name::AbstractString = "reverse AD NLS model",
  ) where {T <: Real, S, R <: Function, AD <: ADBackend}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x, name = name)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    tmp_output = S(undef, nequ)
    return new{T, S, R, AD}(meta, nls_meta, NLSCounters(), r!, tmp_output, rd)
  end
end

# convenience constructor
function ReverseADNLSModel(r!, nequ::Int, x::S; kwargs...) where {S}

  T = eltype(S)
  nvar = length(x)
  rd = ReverseDiffAD(r!, T, nvar, nequ)

  ReverseADNLSModel{T, S, typeof(r!), typeof(rd)}(
    r!,
    nequ,
    x,
    rd;
    kwargs...,
  )
end

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
  jprod_residual!(Jv, nls.rd, x, v, nls._tmp_output)
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
  jtprod_residual!(Jtv, nls.rd, x, v)
end

end # module