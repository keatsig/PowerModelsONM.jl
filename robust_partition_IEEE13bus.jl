## Load packages
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JuMP

## Parse data
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
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

## Generate scenarios
N = 5      # number of scenarios
ΔL = 0.1   # load variability around base value
load_scenarios = generate_load_scenarios(math, N, ΔL)

## Solve robust mld problem
results_robust = ONM.solve_robust_block_mld(math, PMD.LPUBFDiagPowerModel, solver, load_scenarios)