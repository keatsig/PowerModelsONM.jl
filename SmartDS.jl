## Load packages
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JuMP
import JSON

## Parse data
case_name = "C:/Users/358598/Downloads/Repositories/DynaGrid_Wildfire-main/Data/Grid_Data/p17uhs_13/Master_primary.JSON"
eng = PMD.parse_json(case_name)

for (i_tr,tr) in eng["transformer"]
    nwindings = length(tr["bus"])
    tr["tm_nom"] = Base.convert(Vector{Real}, ones(nwindings))
    tr["connections"] = Vector{Int}[]
    for n=1:nwindings
        push!(tr["connections"], eng["bus"][tr["bus"][n]]["terminals"])
    end
	# Remove f_bus and t_bus to avoid interpretation as AL2W transformers
	delete!(tr, "f_bus")
	delete!(tr, "t_bus")
end
eng["switch_close_actions_ub"] = Inf

# Added this to avoid error with vbases_default being the wrong type
eng["settings"]["vbases_default"] = Base.convert(Dict{String, Real}, eng["settings"]["vbases_default"])
PMD.apply_voltage_bounds!(eng)
math = ONM.transform_data_model(eng)

## Solver instance
solver = JuMP.optimizer_with_attributes(
            HiGHS.Optimizer,
            "presolve"=>"on",
            "primal_feasibility_tolerance"=>1e-6,
            "dual_feasibility_tolerance"=>1e-6,
            "mip_feasibility_tolerance"=>1e-4,
            "mip_rel_gap"=>1e-4,
            "small_matrix_value"=>1e-8,
            "allow_unbounded_or_infeasible"=>true,
            "log_to_console"=>false,
            "output_flag"=>false
        )

## Scenario details
N = 2    # number of scenarios
ΔL = 0.1   # load variability around base value

## Solve robust mld problem
results = ONM.solve_robust_block_mld(eng, PMD.LPUBFDiagPowerModel,solver; N, ΔL)

## Rank partitions
partitions = ONM.generate_ranked_robust_partitions(math, results)

