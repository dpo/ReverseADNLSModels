using LinearAlgebra

using BenchmarkTools

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
resid!(rx, x) = BundleAdjustmentModels.residuals!(cams_indices, pnts_indices, x, rx, nobs, npnts)
rd_bamodel = ReverseADNLSModel(resid!, nequ, x0)

rx = Vector{T}(undef, nequ)
@benchmark residual!($rd_bamodel, $x0, $rx)

# make this method general enough for AD
function BundleAdjustmentModels.scaling_factor(point, k1, k2)
  sq_norm_point = dot(point, point)
  return 1 + sq_norm_point * (k1 + k2 * sq_norm_point)
end

v = fill!(similar(x0), 1)
Jv = similar(rx)
@benchmark jprod_residual!($rd_bamodel, $x0, $v, $Jv)

u = fill!(similar(Jv), 1)
Jtu = similar(x0)
@benchmark jtprod_residual!($rd_bamodel, $x0, $u, $Jtu)
