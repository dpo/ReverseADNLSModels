using Test
using NLPModels, ReverseADNLSModels

@testset "basic" begin
  F!(Fx, x) = begin
    Fx[1] = x[1] - 1.0
    Fx[2] = 10 * (x[2] - x[1]^2)
    Fx
  end
  x0 = [-1.2; 1.0]
  nvar = 2
  nequ = 2
  model = ReverseADNLSModel(F!, nequ, x0, name = "bogus NLS")

  Fx = similar(x0)
  residual!(model, x0, Fx)
  @test Fx ≈ [-2.2, -4.4]

  v = ones(nvar)
  Jv = similar(v)
  jprod_residual!(model, x0, v, Jv)
  @test Jv ≈ [1, 34]

  u = ones(nequ)
  Jtu = similar(u)
  jtprod_residual!(model, x0, u, Jtu)
  @test Jtu ≈ [25, 10]
end