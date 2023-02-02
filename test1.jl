using JuMP, HiGHS
solver = optimizer_with_attributes(
	HiGHS.Optimizer,
	"presolve"=>"off",
	"primal_feasibility_tolerance"=>1e-6,
	"dual_feasibility_tolerance"=>1e-6,
	"mip_feasibility_tolerance"=>1e-4,
	"mip_rel_gap"=>1e-4,
	"small_matrix_value"=>1e-12,
	"allow_unbounded_or_infeasible"=>true,
    "output_flag"=>true,
)

N=2
for scenario=1:N
    local model = Model()
    set_optimizer(model, solver)

    global z_block = JuMP.@variable(
        model,
        [i in 1:5],
        base_name="0_z_block_$(scen)",
        lower_bound=0,
        upper_bound=1,
        binary=true)

    # @objective(model, Max, sum(sum(z_block[scen][i] for i in 1:5) for scen in 1:scenario))
    optimize!(model);

    for scen in 1:scenario
        # @show [value.(z_block[scen][i] for i in 1:5)]
    end
end

using JuMP, Clp, Ipopt

m = Model(Clp.Optimizer)
set_A = [(1,2), (2,3)]
x = @variable(m, [(i,j) in set_A], base_name="x_1")
@constraint(m, x.>=0)
@objective(m, Max, sum(x[(i,j)] for (i,j) in set_A) )
@constraint(m, x[(1,2)]<=10)
@constraint(m, x[(2,3)]<=10)
optimize!(m)
@show value.(x)

m2,ref = copy_model(m)
set_optimizer(m2,Clp.Optimizer)
x1 = ref[x]
x2 = @variable(m, [(i,j) in set_A], base_name="x_2")
@constraint(m, x2.>=0)
@constraint(m, x2[(1,2)]<=1)
@constraint(m, x2[(2,3)]<=1)
@objective(m, Max, sum(x2[(i,j)]-x[(i,j)] for (i,j) in set_A) )
optimize!(m)
@show value.(x)
@show value.(x2)

# m = Model(Clp.Optimizer)
# x = @variable(m, base_name="x")
# y = @variable(m, base_name="x")
# @constraint(m, x>=0)
# @constraint(m, y>=2)
# @constraint(m, x<=2)
# @constraint(m, y<=4)
# optimize!(m)
# @show value.(x)
# @show value.(y)
