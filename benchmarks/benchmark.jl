using LinearAlgebra

using BenchmarkTools
using ForwardDiff
using ReverseDiff
using SparseDiffTools

using BundleAdjustmentModels
using NLPModels
using ReverseADNLSModels

df = problems_df()
filter_df = df[ ( df.nequ .≥ 50000 ) .& ( df.nvar .≤ 34000 ), :]
name, group = get_first_name_and_group(filter_df)
path = fetch_ba_name(name, group)
filename = BundleAdjustmentModels.get_filename(name, group)
filedir = BundleAdjustmentModels.fetch_ba_name(filename, group)
path_and_filename = joinpath(filedir, filename)
T = Float64
cams_indices, pnts_indices, pt2d, x0, ncams, npnts, nobs = BundleAdjustmentModels.readfile(path_and_filename, T = T)
nvar = 9 * ncams + 3 * npnts
nequ = 2 * nobs

# residual function for evaluation
k = similar(x0)
P1 = similar(x0)
resid!(rx, x) = BundleAdjustmentModels.residuals!(cams_indices, pnts_indices, x, rx, nobs, npnts, k, P1)

# residual function for forward AD
k_fwd = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
P1_fwd = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
resid_fwd!(rx, x) = BundleAdjustmentModels.residuals!(cams_indices, pnts_indices, x, rx, nobs, npnts, k_fwd, P1_fwd)

# residual function for reverse AD
k_rev = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
P1_rev = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
resid_rev!(rx, x) = BundleAdjustmentModels.residuals!(cams_indices, pnts_indices, x, rx, nobs, npnts, k_rev, P1_rev)

rd_bamodel = ReverseADNLSModel(resid!, nequ, x0, r_fwd! = resid_fwd!, r_rev! = resid_rev!)

@info "benchmarking residual"
rx = Vector{T}(undef, nequ)
b_resid = @benchmark residual!($rd_bamodel, $x0, $rx)
display(b_resid)

@info "benchmarking jprod"
# make this method general enough for AD
function BundleAdjustmentModels.scaling_factor(point, k1, k2)
  sq_norm_point = dot(point, point)
  return 1 + sq_norm_point * (k1 + k2 * sq_norm_point)
end

v = fill!(similar(x0), 1)
Jv = similar(rx)
b_jprod = @benchmark jprod_residual!($rd_bamodel, $x0, $v, $Jv)
display(b_jprod)

@info "benchmarking jtprod"
u = fill!(similar(Jv), 1)
Jtu = similar(x0)
b_jtprod = @benchmark jtprod_residual!($rd_bamodel, $x0, $u, $Jtu)
display(b_jtprod)
