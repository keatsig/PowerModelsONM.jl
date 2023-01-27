import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import HiGHS
import JSON
import JuMP

onm_path = joinpath(dirname(pathof(ONM)), "..")
eng_s = ONM.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
PMD.apply_voltage_bounds!(eng_s)
# settings = ONM.parse_settings(joinpath(onm_path,"test/data/ieee13_settings.json"))

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

# eng_s = ONM.apply_settings(eng_s, settings)

# eng_s["time_elapsed"] = 1.0

contingencies = ONM.generate_n_minus_contingencies(eng_s, 6)

results = ONM.generate_robust_partitions(eng_s, contingencies, PMD.LPUBFDiagPowerModel, solver)

parts = ONM.generate_ranked_robust_partitions(eng_s, results)

open("robust_partitions_ieee13.json", "w") do io
    JSON.print(io, parts, 2)
end

load_scenarios = ONM.generate_load_scenarios(eng_s, 5, 0.1)
eng_dict = Dict(); idx = 1
results = Dict{String,Any}()
results_robust = Dict{String,Any}()
for state in contingencies
    eng = deepcopy(eng_s)
    eng["switch"] = recursive_merge(get(eng, "switch", Dict{String,Any}()), state)
    eng["time_elapsed"] = 1.0  # TODO: what should the time_elapsed be? how long do we want partitions to be robust for?
    eng_dict[idx] = eng
    # results["$(idx)"] = ONM.solve_robust_partitions(eng, PMD.LPUBFDiagPowerModel, solver)
    results["$(idx)"] = ONM.solve_robust_block_mld(eng, PMD.LPUBFDiagPowerModel,solver, load_scenarios)
    idx += 1
end

eng1 = eng_dict[2]

results = ONM.solve_robust_partitions(eng1, PMD.LPUBFDiagPowerModel, solver)
results_robust = ONM.solve_robust_block_mld(eng1, PMD.LPUBFDiagPowerModel,solver, load_scenarios)
for (id,result) in results_robust["[1]"]["solution"]["switch"]
    @show data_math["switch"][id]["source_id"], result["state"]
end

results = generate_load_robust_partitions(eng_s, contingencies, PMD.LPUBFDiagPowerModel, solver; N=5, ΔL=0.1)
data = ONM.transform_data_model(eng_s)

robust_partitions, robust_partitions_uncertainty = ONM.generate_ranked_robust_partitions_with_uncertainty(data, results)
open("robust_partitions.json", "w") do io
    JSON.print(io, robust_partitions, 2)
end
x=[]
for (id,result) in results
    push!(x,result["termination_status"])
end

xx=[]
for (id,result) in results_robust
    push!(xx,result["[1]"]["termination_status"])
end

for (id,result) in results_robust
    @show id,length(result)
end

for (id,result) in results["solution"]["switch"]
    @show id, results["solution"]["switch"][id]["state"]
end


for (id,result) in results_robust["[1]"]["solution"]["switch"]
    @show data_math["switch"][id]["source_id"], result["state"]
end
