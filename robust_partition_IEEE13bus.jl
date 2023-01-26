## Load packages
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JuMP

## Parse data
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng_s = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
settings = ONM.parse_settings(joinpath(onm_path,"test/data/ieee13_settings.json"))
eng_s = ONM.apply_settings(eng_s, settings)
eng_s["time_elapsed"] = 1.0
eng_s["switch_close_actions_ub"] = 1
PMD.apply_voltage_bounds!(eng_s)
math = ONM.transform_data_model(eng_s)

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
N = 3     # number of scenarios
ΔL = 0.1   # load variability around base value

## Solve robust mld problem
results_robust = ONM.solve_robust_block_mld(math, PMD.LPUBFDiagPowerModel,solver; N, ΔL)

## Rank partitions
partitions = ONM.update_ranked_robust_partitions(math, results_robust)

