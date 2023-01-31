
import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JSON
import JuMP

onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))

settings = ONM.parse_settings(joinpath(onm_path,"test/data/ieee13_settings.json"))

solver = ONM.optimizer_with_attributes(
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

eng_s = ONM.apply_settings(eng, settings)

eng_s["time_elapsed"] = 1.0

contingencies = ONM.generate_n_minus_contingencies(eng_s, 6)

results = ONM.generate_robust_partitions(eng_s, contingencies, PMD.LPUBFDiagPowerModel, solver)

parts = ONM.generate_ranked_robust_partitions(eng_s, results)

open("draft_robust_partitions_ieee13.json", "w") do io
    JSON.print(io, parts, 2)
end
