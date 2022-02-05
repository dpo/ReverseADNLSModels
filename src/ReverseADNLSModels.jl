module ReverseADNLSModels
using LinearAlgebra
using ForwardDiff, ReverseDiff
using NLPModels

export ReverseADNLSModel

abstract type ADBackend end

# ForwardDiff.derivative!(Jv, ϕ!, Fx, zero(eltype(x)))
struct ForwardDiffAD{F, T} <: ADBackend where {F <: Function, T <: Real}
  ϕ!::F
  tmp_out::Vector{T}
end

function ForwardDiffAD(r!::R, T::DataType, nequ::Int) where R <: Function
  ϕ!(out, t, x, v) = begin
    r!(out, x + t * v)  # no idea how to preallocate a vector for x + t * v ???
    out
  end
  # can't figure out how to do jtprod without allocating...
  tmp_out = Vector{T}(undef, nequ)
  ForwardDiffAD{typeof(ϕ!), T}(ϕ!, tmp_out)
end

function jprod_residual!(Jv, fd::ForwardDiffAD{F,T}, x, v, args...) where {F <: Function, T <: Real}
  ForwardDiff.derivative!(Jv, (out, t) -> fd.ϕ!(out, t, x, v), fd.tmp_out, 0)
  Jv
end

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
mutable struct ReverseADNLSModel{T, S, R, AD1, AD2} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  _tmp_output::S
  fd::AD1
  rd::AD2
  adbackends::Dict{Symbol, ADBackend}

  function ReverseADNLSModel{T, S, R, AD1, AD2}(
    r!::R,
    nequ::Int,
    x::S,
    fd::AD1,
    rd::AD2;
    name::AbstractString = "reverse AD NLS model",
  ) where {T <: Real, S, R <: Function, AD1 <: ADBackend, AD2 <: ADBackend}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x, name = name)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    tmp_output = S(undef, nequ)
    adbackends = Dict{Symbol, ADBackend}(:jprod_residual! => fd, :jtprod_residual! => rd)
    return new{T, S, R, AD1, AD2}(meta, nls_meta, NLSCounters(), r!, tmp_output, fd, rd, adbackends)
  end
end

# convenience constructor
function ReverseADNLSModel(r!, nequ::Int, x::S; kwargs...) where {S}

  T = eltype(S)
  nvar = length(x)
  fd = ForwardDiffAD(r!, T, nequ)
  rd = ReverseDiffAD(r!, T, nvar, nequ)

  ReverseADNLSModel{T, S, typeof(r!), typeof(fd), typeof(rd)}(
    r!,
    nequ,
    x,
    fd,
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
  jprod_residual!(Jv, nls.adbackends[:jprod_residual!], x, v, nls._tmp_output)
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
  jtprod_residual!(Jtv, nls.adbackends[:jtprod_residual!], x, v)
end

end # module