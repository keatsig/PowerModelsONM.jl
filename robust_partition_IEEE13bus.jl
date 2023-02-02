## Load packages
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JuMP

## Parse data
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
# PMD.apply_voltage_bounds!(eng)
settings = ONM.parse_settings("test/data/ieee13_settings.json")
settings["options"]["data"]["switch-close-actions-ub"] = Inf
eng = ONM.apply_settings(eng, settings)
eng["time_elapsed"] = 1.0
math = ONM.transform_data_model(eng)

eng = PMD.parse_file(joinpath(onm_path, "test/data/ieee13_feeder1.dss"); transformations=[remove_line_limits!])
PMD.apply_voltage_bounds!(eng)
eng["switch_close_actions_ub"] = Inf

eng = parse_json("C:/Users/358598/Downloads/Repositories/DynaGrid_Wildfire/Data/Grid_Data/p17uhs_13/Master_primary.json")
for (i_tr,tr) in eng["transformer"]
    nwindings = length(tr["bus"])
    tr["tm_nom"] = convert(Vector{Real}, ones(nwindings))
    tr["connections"] = Vector{Int}[]
    for n=1:nwindings
        push!(tr["connections"], eng["bus"][tr["bus"][n]]["terminals"])
    end
	# Remove f_bus and t_bus to avoid interpretation as AL2W transformers
	delete!(tr, "f_bus")
	delete!(tr, "t_bus")
end
eng["switch_close_actions_ub"] = Inf
eng["settings"]["vbases_default"] = convert(Dict{String, Real}, eng["settings"]["vbases_default"])
apply_voltage_bounds!(eng)
math = transform_data_model(eng)

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

## Generate scenarios
N = 100      # number of scenarios
ΔL = 0.5   # load variability around base value
load_scenarios = ONM.generate_load_scenarios(eng, N, ΔL)

## Solve robust mld problem
results_robust = ONM.solve_robust_block_mld(eng, PMD.LPUBFDiagPowerModel, solver, load_scenarios)
# results_robust_EJ = ONM.solve_robust_block_mld(eng, PMD.LPUBFDiagPowerModel, solver, load_scenarios)
