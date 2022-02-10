module ReverseADNLSModels
using LinearAlgebra
using ForwardDiff, ReverseDiff, SparseDiffTools
using NLPModels

export ReverseADNLSModel

abstract type ADBackend end

struct ForwardDiffAD{T, F} <: ADBackend where {T, F <: Function}
  r!::F
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end

function ForwardDiffAD(r!::F, T::DataType, nvar::Int, nequ::Int) where {F <: Function}
  tmp_in = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nequ)
  ForwardDiffAD{T, F}(r!, tmp_in, tmp_out)
end

function jprod_residual!(Jv::AbstractVector{T}, fd::ForwardDiffAD{T}, x::AbstractVector{T}, v::AbstractVector{T}, args...) where T
  SparseDiffTools.auto_jacvec!(Jv, fd.r!, x, v, fd.tmp_in, fd.tmp_out)
  Jv
end

struct ReverseDiffAD{T, F1, F2} <: ADBackend where {T, F1 <: Function, F2 <: Function}
  ϕ!::F1
  ψ::F2
  _tmp_input::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _tmp_output::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  z::Vector{T}
end

function ReverseDiffAD(r!::F, T::DataType, nvar::Int, nequ::Int) where {F <: Function}
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

struct ReverseDiffADTape{T, F2, GC, GT} <: ADBackend where {T, F2 <: Function, GC, GT}
  # ϕ!::F1
  ψ::F2
  gcfg::GC  # gradient config
  gtape::GT  # compiled gradient tape
  _tmp_output::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _rval::Vector{T}  # temporary storage for jtprod
end

function ReverseDiffADTape(r!::F, x0::AbstractArray{T}, nequ::Int) where {T, F <: Function}
  _tmp_output_rd = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nequ)
  _ψ(x, u, tmp_out) = begin
    # here x is a vector of ReverseDiff.TrackedReal
    r!(tmp_out, x)
    dot(tmp_out, u)
  end
  ψ = (x, u) -> _ψ(x, u, _tmp_output_rd)
  u = similar(x0, nequ)  # just for GradientConfig
  gcfg = ReverseDiff.GradientConfig((x0, u))
  gtape = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (x0, u), gcfg))
  rval = similar(x0, nequ)  # temporary storage for jtprod
  ReverseDiffADTape{T, typeof(ψ), typeof(gcfg), typeof(gtape)}(ψ, gcfg, gtape, _tmp_output_rd, rval)
end

function jtprod_residual!(Jtv, rd::ReverseDiffADTape{T, F2, GC, GT}, x, v) where {T, F2 <: Function, GC, GT}
  ReverseDiff.gradient!((Jtv, rd._rval), rd.gtape, (x, v))
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
# TODO: Explain why r_fwd! and r_rev! might differ from r!
function ReverseADNLSModel(r!, nequ::Int, x::S; r_fwd! = r!, r_rev! = r!, kwargs...) where {S}

  T = eltype(S)
  nvar = length(x)
  fd = ForwardDiffAD(r_fwd!, T, nvar, nequ)
  # rd = ReverseDiffAD(r_rev!, T, nvar, nequ)
  rd = ReverseDiffADTape(r_rev!, x, nequ)

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