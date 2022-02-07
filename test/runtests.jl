using Test
using NLPModels, ReverseADNLSModels

@testset "basic" begin
  F!(Fx, x) = begin
    Fx[1] = x[1] - 1
    Fx[2] = 10 * (x[2] - x[1]^2)
    Fx[3] = x[2] + 1
    Fx
  end
  x0 = [-1.2; 1.0]
  nvar = 2
  nequ = 3
  model = ReverseADNLSModel(F!, nequ, x0, name = "bogus NLS")

  Fx = similar(x0, nequ)
  residual!(model, x0, Fx)
  @test Fx ≈ [-2.2, -4.4, 2.0]

  v = ones(nvar)
  Jv = similar(v, nequ)
  jprod_residual!(model, x0, v, Jv)
  @test Jv ≈ [1, 34, 1]

  nallocs_jprod = @allocated jprod_residual!(model, x0, v, Jv)
  @test nallocs_jprod == 0

  u = ones(nequ)
  Jtu = similar(u, nvar)
  jtprod_residual!(model, x0, u, Jtu)
  @test Jtu ≈ [25, 11]
end