
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

eng_dict = Dict(); idx = 1
load_scenarios = ONM.generate_load_scenarios(eng_s, 3, 0.1)

 for state in contingencies
    eng = deepcopy(eng_s)
    eng["switch"] = recursive_merge(get(eng, "switch", Dict{String,Any}()), state)
    # @show eng["switch"] == recursive_merge(get(eng, "switch", Dict{String,Any}()), state)
    eng["time_elapsed"] = 1.0  # TODO: what should the time_elapsed be? how long do we want partitions to be robust for?
    eng["switch_close_actions_ub"] = 1
    eng_dict[idx] = eng
    idx += 1
end
eng1 = eng_dict[2]
data_math = ONM.transform_data_model(eng1)
results_robust = ONM.solve_robust_block_mld(eng1, PMD.LPUBFDiagPowerModel,solver, load_scenarios)

generate_load_robust_partitions(eng_s, contingencies, PMD.LPUBFDiagPowerModel, solver; N=2, ΔL=0.1)


for (id,result) in results
    @show result["termination_status"]
end
