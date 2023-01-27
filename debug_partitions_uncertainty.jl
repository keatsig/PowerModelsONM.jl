## Load packages
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JSON
import JuMP

## Parse data
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
eng["switch_close_actions_ub"] = Inf
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

## Generate contingencies
contingencies = ONM.generate_n_minus_contingencies(eng, 6)

## Generate scenarios
N = 5      # number of scenarios
ΔL = 0.1   # load variability around base value
load_scenarios = ONM.generate_load_scenarios(eng, N, ΔL)

## Solve robust partition problem considering uncertainty for all contingencies
results = ONM.generate_load_robust_partitions(eng, contingencies, load_scenarios, PMD.LPUBFDiagPowerModel, solver)

## Rank partitions
robust_partitions, robust_partitions_uncertainty = ONM.generate_ranked_robust_partitions_with_uncertainty(math, results)

## Save robust partitions (without considering uncertainty)
open("robust_partitions.json", "w") do io
    JSON.print(io, robust_partitions, 2)
end

## Save robust partitions (considering uncertainty)
open("robust_partitions_uncertainty.json", "w") do io
    JSON.print(io, robust_partitions_uncertainty, 2)
end
