## Load packages
using Revise, PowerModelsDistribution, PowerModelsONM
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JSON
import JuMP
import JLD
# import Gurobi
PMD.silence!()

## Parse data
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
settings = ONM.parse_settings("test/data/ieee13_settings.json")
settings["options"]["data"]["switch-close-actions-ub"] = Inf
eng = ONM.apply_settings(eng, settings)
eng["time_elapsed"] = 1.0

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
# solver = JuMP.optimizer_with_attributes(
#     () -> ONM.GRB_ENV,
#     "presolve"=>"on",
#     "primal_feasibility_tolerance"=>1e-6,
#     "dual_feasibility_tolerance"=>1e-6,
#     "mip_feasibility_tolerance"=>1e-4,
#     "mip_rel_gap"=>1e-4,
#     "small_matrix_value"=>1e-8,
#     "allow_unbounded_or_infeasible"=>true,
#     "log_to_console"=>false,
#     "output_flag"=>false
# )

## Generate contingencies
contingencies = ONM.generate_n_minus_contingencies(eng, 6)

## Generate load scenarios
N = 1     # number of scenarios
ΔL = 0.1  # load variability around base value
load_scenarios = ONM.generate_load_scenarios(eng, N, ΔL)

## Solve robust partition problem considering uncertainty for all contingencies (check if line 251 in prob/mld_block_robust.jl is commented OUT)
results = ONM.generate_load_robust_partitions(eng, contingencies, load_scenarios, PMD.LPUBFDiagPowerModel, solver)

#--------------------------RUN TILL HERE AND STOP--------------------------------------#
#-----comment out line 250 and comment in line 251 in prob/mld_block_robust.jl---------#
#-----add EJ metrics data in objective_robust_partition_EJ() in core\objective.jl------#

## Solve robust partition problem with EJ objective considering uncertainty for all contingencies (check if line 251 in prob/mld_block_robust.jl is commented IN)
results_EJ = ONM.generate_load_robust_partitions(eng, contingencies, load_scenarios, PMD.LPUBFDiagPowerModel, solver)

## Rank partitions
robust_partitions, robust_partitions_uncertainty = ONM.generate_ranked_robust_partitions_with_uncertainty(eng, results)
robust_partitions_EJ, robust_partitions_uncertainty_EJ = ONM.generate_ranked_robust_partitions_with_uncertainty(eng, results_EJ)

## Save robust partition rankings (no load uncertainty, no EJ metrics)
open("ex_robust_partitions.json", "w") do io
    JSON.print(io, robust_partitions, 2)
end

## Save robust partition rankings (with load uncertainty, no EJ metrics)
open("ex_robust_partitions_uncertainty.json", "w") do io
    JSON.print(io, robust_partitions_uncertainty, 2)
end

## Save robust partition rankings (no load uncertainty, with EJ metrics)
open("ex_robust_partitions_EJ.json", "w") do io
    JSON.print(io, robust_partitions_EJ, 2)
end

## Save robust partition rankings (with load uncertainty, with EJ metrics)
open("ex_robust_partitions_uncertainty_EJ.json", "w") do io
    JSON.print(io, robust_partitions_uncertainty_EJ, 2)
end
