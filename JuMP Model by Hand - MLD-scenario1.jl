import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import InfrastructureModels as IM
import JuMP
import Ipopt
import HiGHS
import LinearAlgebra
import StatsBase as SB
import JLD
PMD.silence!()

## build model and initialize constant parameters
onm_path = joinpath(dirname(pathof(ONM)), "..")
case_file = joinpath(onm_path, "test/data/ieee13_feeder.dss")

global eng = PMD.parse_file(case_file)
eng["switch_close_actions_ub"] = Inf
PMD.apply_voltage_bounds!(eng)
global math = ONM.transform_data_model(eng)
global ref = IM.build_ref(
    math,
    PMD.ref_add_core!,
    union(ONM._default_global_keys, PMD._pmd_math_global_keys),
    PMD.pmd_it_name;
    ref_extensions=ONM._default_ref_extensions
)[:it][:pmd][:nw][IM.nw_id_default]

# branch parameters
global branch_connections = Dict((l,i,j) => connections for (bus,entry) in ref[:bus_arcs_conns_branch] for ((l,i,j), connections) in entry)

# switch parameters
global switch_arc_connections = Dict((l,i,j) => connections for (bus,entry) in ref[:bus_arcs_conns_switch] for ((l,i,j), connections) in entry)
global switch_close_actions_ub = ref[:switch_close_actions_ub]

# transformer parameters
global transformer_connections = Dict((l,i,j) => connections for (bus,entry) in ref[:bus_arcs_conns_transformer] for ((l,i,j), connections) in entry)
global p_oltc_ids = [id for (id,trans) in ref[:transformer] if !all(trans["tm_fix"])]

# load parameters
global load_wye_ids = [id for (id, load) in ref[:load] if load["configuration"]==PMD.WYE]
global load_del_ids = [id for (id, load) in ref[:load] if load["configuration"]==PMD.DELTA]
global load_cone_ids = [id for (id, load) in ref[:load] if PMD._check_load_needs_cone(load)]
global load_connections = Dict{Int,Vector{Int}}(id => load["connections"] for (id,load) in ref[:load])

# grid-forming inverter parameters
global L = Set(keys(ref[:blocks]))
global map_id_pairs = Dict(id => (ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (id,sw) in ref[:switch])
global Φₖ = Dict(k => Set() for k in L)
global map_virtual_pairs_id = Dict(k=>Dict() for k in L)
for kk in L # color
    touched = Set()
    ab = 1
    for k in sort(collect(L)) # fr block
        for k′ in sort(collect(filter(x->x!=k,L))) # to block
            if (k,k′) ∉ touched
                map_virtual_pairs_id[kk][(k,k′)] = map_virtual_pairs_id[kk][(k′,k)] = ab
                push!(touched, (k,k′), (k′,k))
                ab += 1
            end
        end
    end
    Φₖ[kk] = Set([map_virtual_pairs_id[kk][(kk,k′)] for k′ in filter(x->x!=kk,L)])
end

# storage parameters
global storage_inj_lb, storage_inj_ub
storage_inj_lb, storage_inj_ub = PMD.ref_calc_storage_injection_bounds(ref[:storage], ref[:bus])

# topology parameters
global _N₀ = collect(keys(ref[:blocks]))
global _L₀ = ref[:block_pairs]
global virtual_iᵣ = maximum(_N₀)+1
global _N = [_N₀..., virtual_iᵣ]
global iᵣ = [virtual_iᵣ]
global _L = [_L₀..., [(virtual_iᵣ, n) for n in _N₀]...]
global _L′ = union(_L, Set([(j,i) for (i,j) in _L]))

# objective parameters
global total_energy_ub = sum(strg["energy_rating"] for (i,strg) in ref[:storage])
global total_pmax = sum(Float64[all(.!isfinite.(gen["pmax"])) ? 0.0 : sum(gen["pmax"][isfinite.(gen["pmax"])]) for (i, gen) in ref[:gen]])
global total_energy_ub = total_energy_ub <= 1.0 ? 1.0 : total_energy_ub
global total_pmax = total_pmax <= 1.0 ? 1.0 : total_pmax
global n_dispatchable_switches = length(keys(ref[:switch_dispatchable]))
global n_dispatchable_switches = n_dispatchable_switches < 1 ? 1 : n_dispatchable_switches
global block_weights = ref[:block_weights]

# solver instance setup
global solver = JuMP.optimizer_with_attributes(
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

## setup scenario model, solve and check feasibility
function solve_model(N_scen::Int, ΔL::Float64)

    # Generate scenarios
    load_factor = generate_load_scenarios(math, N_scen, ΔL)
    # load_factor = JLD.load("load_factor.jld")["load_factor"]

    # create empty model and generate common variables
    model = JuMP.Model()
    JuMP.set_optimizer(model, solver)
    all_var_common = Dict()
    variable_common_model(model,all_var_common)

    # setup and solve model adding one scenario in each iteration
    all_var_scen = Dict(scen=> Dict() for scen=1:N_scen)
    all_var_common_soln = Dict()
    scenarios = [1]
    idx = 0
    viol_ind = true
    while length(scenarios)<=N_scen && viol_ind

        idx += 1

        for scen in scenarios[idx:end]
            # add variables to model
            variable_model(model,all_var_scen[scen],scen,load_factor[scen])

            # add constraints to model
            constraint_model(model,all_var_scen[scen],all_var_common,load_factor[scen])
        end

        # objective
        JuMP.@objective(model, Min, sum(
                sum( block_weights[i] * (1-all_var_scen[scen]["z_block"][i]) for (i,block) in ref[:blocks])
                + sum( ref[:switch_scores][l]*(1-all_var_common["z_switch"][l]) for l in keys(ref[:switch_dispatchable]) )
                + sum( all_var_scen[scen]["delta_sw_state"][l] for l in keys(ref[:switch_dispatchable])) / n_dispatchable_switches
                + sum( (strg["energy_rating"] - all_var_scen[scen]["se"][i]) for (i,strg) in ref[:storage]) / total_energy_ub
                + sum( sum(get(gen,  "cost", [0.0, 0.0])[2] * all_var_scen[scen]["pg"][i][c] + get(gen,  "cost", [0.0, 0.0])[1] for c in  gen["connections"]) for (i,gen) in ref[:gen]) / total_energy_ub
        for scen in scenarios) )

        # solve manual model
        JuMP.optimize!(model)

        # print output
        obj_val = []
        for scen in scenarios
            obj_scen = sum( block_weights[i] * (1-all_var_scen[scen]["z_block"][i]) for (i,block) in ref[:blocks])+ sum( ref[:switch_scores][l]*(1-all_var_common["z_switch"][l]) for l in keys(ref[:switch_dispatchable]) )+ sum( all_var_scen[scen]["delta_sw_state"][l] for l in keys(ref[:switch_dispatchable])) / n_dispatchable_switches+ sum( (strg["energy_rating"] - all_var_scen[scen]["se"][i]) for (i,strg) in ref[:storage]) / total_energy_ub+ sum( sum(get(gen,  "cost", [0.0, 0.0])[2] * all_var_scen[scen]["pg"][i][c] + get(gen,  "cost", [0.0, 0.0])[1] for c in  gen["connections"]) for (i,gen) in ref[:gen]) / total_energy_ub
            push!(obj_val,round(JuMP.value(obj_scen), digits=4))
        end
        sts = string(JuMP.termination_status(model))
        println("$(scenarios): $(sts) Obj_val=$(obj_val)")
        println("Switch status: $([JuMP.value(all_var_common["z_switch"][i]) for i in keys(ref[:switch_dispatchable])])")
        println("Inverter status: $([JuMP.value(z_inv[i]) for ((t,i), z_inv) in all_var_common["z_inverter"]])")
        # if scenario == N_scen
            for scen in scenarios
                println("Substation pg for scenario $(scen): $([JuMP.value(all_var_scen[scen]["pg"][4][c]) for c=1:3])")
                println("Storage se for scenario $(scen): $([JuMP.value(all_var_scen[scen]["se"][i]) for (i,strg) in ref[:storage]])")
                # println("z_block for scenario $(scen): $([JuMP.value(all_var_scen[scen]["z_block"][c]) for c in keys(ref[:blocks])])")
            end
        # end
        all_var_common_soln["z_inverter"] = Dict(
            (t,i) => JuMP.value(z_inv) for ((t,i), z_inv) in all_var_common["z_inverter"]
        )
        all_var_common_soln["z_switch"] = Dict(i => JuMP.value(all_var_common["z_switch"][i]) for i in keys(ref[:switch_dispatchable]))

        # feasibility check
        scenario = deleteat!([1:N_scen;], sort(scenarios))
        if length(scenario)==0
            viol_ind = false
        else
            infeas_idx = feasibility_check(scenario,all_var_common_soln,load_factor)
            if length(infeas_idx) > 0
                for idx in infeas_idx
                    push!(scenarios,idx)
                end
            else
                viol_ind = false
            end
        end
    end

end

## generate load scenarios
function generate_load_scenarios(data::Dict{String,<:Any}, N::Int, ΔL::Float64)
    if PMD.iseng(data)
        data = ONM.transform_data_model(data)
    end
    n_l = length(data["load"])
    load_factor = Dict(scen => Dict() for scen in 1:N)
    scen = 1
    while scen<=N
        data_scen = deepcopy(data)
        uncertain_scen = SB.sample((1-ΔL):(2*ΔL/n_l):(1+ΔL), n_l, replace=false)
        for (id,load) in data["load"]
            if scen==1
                load_factor[1][id] = 1
            else
                load_factor[scen][id] = uncertain_scen[parse(Int64,id)]
                data_scen["load"][id]["pd"] = load["pd"]*uncertain_scen[parse(Int64,id)]
                data_scen["load"][id]["qd"] = load["qd"]*uncertain_scen[parse(Int64,id)]
            end
        end
        result = PMD.solve_mc_opf(data_scen, PMD.LPUBFDiagPowerModel, JuMP.optimizer_with_attributes(Ipopt.Optimizer,"print_level"=>0))
        if string(result["termination_status"]) == "LOCALLY_SOLVED"
            scen += 1
        end
    end

    # save load_factor to file
    JLD.save("load_factor.jld", "load_factor", load_factor)

    return load_factor
end


## add variables for each scenario
function variable_model(model::JuMP.Model, var_scen::Dict{Any,Any}, scen::Int, load_factor_scen::Dict{Any, Any}; feas_chck::Bool=false)

    # variable_block_indicator
    var_scen["z_block"] = JuMP.@variable(
        model,
        [i in keys(ref[:blocks])],
        base_name="0_z_block_$(scen)",
        lower_bound=0,
        upper_bound=1,
        binary=true
    )

    # variable_mc_bus_voltage_on_off -> variable_mc_bus_voltage_magnitude_sqr_on_off
    var_scen["w"] = Dict(
        i => JuMP.@variable(
            model,
            [t in bus["terminals"]],
            base_name="0_w_$(i)_$(scen)",
            lower_bound=0,
        ) for (i,bus) in ref[:bus]
    )

    # variable_mc_branch_power
    var_scen["p"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in branch_connections[(l,i,j)]],
                base_name="0_p_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_branch]
        )
    )
    var_scen["q"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in branch_connections[(l,i,j)]],
                base_name="0_q_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_branch]
        )
    )

    # variable_mc_switch_power
    var_scen["psw"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in switch_arc_connections[(l,i,j)]],
                base_name="0_psw_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_switch]
        )
    )

    var_scen["qsw"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in switch_arc_connections[(l,i,j)]],
                base_name="0_qsw_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_switch]
        )
    )

    # this explicit type erasure is necessary
    psw_expr_from = Dict( (l,i,j) => var_scen["psw"][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] )
    var_scen["psw_expr"] = merge(psw_expr_from, Dict( (l,j,i) => -1.0.*var_scen["psw"][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from]))
    var_scen["psw_auxes"] = Dict(
        (l,i,j) => JuMP.@variable(
            model, [c in switch_arc_connections[(l,i,j)]],
            base_name="0_psw_aux_$((l,i,j))_$(scen)"
        ) for (l,i,j) in ref[:arcs_switch]
    )

    qsw_expr_from = Dict( (l,i,j) => var_scen["qsw"][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] )
    var_scen["qsw_expr"] = merge(qsw_expr_from, Dict( (l,j,i) => -1.0.*var_scen["qsw"][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from]))
    var_scen["qsw_auxes"] = Dict(
        (l,i,j) => JuMP.@variable(
            model, [c in switch_arc_connections[(l,i,j)]],
            base_name="0_qsw_aux_$((l,i,j))_$(scen)"
        ) for (l,i,j) in ref[:arcs_switch]
    )

    # variable_mc_transformer_power
    var_scen["pt"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in transformer_connections[(l,i,j)]],
                base_name="0_pt_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_transformer]
        )
    )

    var_scen["qt"] = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in transformer_connections[(l,i,j)]],
                base_name="0_qt_($l,$i,$j)_$(scen)"
            ) for (l,i,j) in ref[:arcs_transformer]
        )
    )

    # variable_mc_oltc_transformer_tap
    var_scen["tap"] = Dict(
        i => JuMP.@variable(
            model,
            [p in 1:length(ref[:transformer][i]["f_connections"])],
            base_name="0_tm_$(i)_$(scen)",
        ) for i in keys(filter(x->!all(x.second["tm_fix"]), ref[:transformer]))
    )

    # variable_mc_generator_power_on_off
    var_scen["pg"] = Dict(
        i => JuMP.@variable(
            model,
            [c in gen["connections"]],
            base_name="0_pg_$(i)_$(scen)",
        ) for (i,gen) in ref[:gen]
    )

    var_scen["qg"] = Dict(
        i => JuMP.@variable(
            model,
            [c in gen["connections"]],
            base_name="0_qg_$(i)_$(scen)",
        ) for (i,gen) in ref[:gen]
    )

    # variable_mc_storage_power_on_off and variable_mc_storage_power_control_imaginary_on_off
    var_scen["ps"] = Dict(
        i => JuMP.@variable(
            model,
            [c in ref[:storage][i]["connections"]],
            base_name="0_ps_$(i)_$(scen)",
        ) for i in keys(ref[:storage])
    )

    var_scen["qs"] = Dict(
        i => JuMP.@variable(
            model,
            [c in ref[:storage][i]["connections"]],
            base_name="0_qs_$(i)_$(scen)",
        ) for i in keys(ref[:storage])
    )

    var_scen["qsc"] = JuMP.@variable(
        model,
        [i in keys(ref[:storage])],
        base_name="0_qsc_$(i)_$(scen)"
    )

    # qsc bounds
    for (i,strg) in ref[:storage]
        if isfinite(sum(storage_inj_lb[i])) || haskey(strg, "qmin")
            lb = max(sum(storage_inj_lb[i]), sum(get(strg, "qmin", -Inf)))
            JuMP.set_lower_bound(var_scen["qsc"][i], min(lb, 0.0))
        end
        if isfinite(sum(storage_inj_ub[i])) || haskey(strg, "qmax")
            ub = min(sum(storage_inj_ub[i]), sum(get(strg, "qmax", Inf)))
            JuMP.set_upper_bound(var_scen["qsc"][i], max(ub, 0.0))
        end
   end

    # variable_storage_energy, variable_storage_charge and variable_storage_discharge
    var_scen["se"] = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_se_$(scen)",
        lower_bound = 0.0,
    )

    var_scen["sc"] = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sc_$(scen)",
        lower_bound = 0.0,
    )

    var_scen["sd"] = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sd_$(scen)",
        lower_bound = 0.0,
    )

    # variable_storage_complementary_indicator and variable_storage_complementary_indicator
    var_scen["sc_on"] = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sc_on_$(scen)",
        binary = true,
        lower_bound=0,
        upper_bound=1
    )

    var_scen["sd_on"] = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sd_on_$(scen)",
        binary = true,
        lower_bound=0,
        upper_bound=1
    )

    # load variables
    var_scen["pd"] = Dict()
    var_scen["qd"] = Dict()
    var_scen["pd_bus"] = Dict()
    var_scen["qd_bus"] = Dict()

    for i in intersect(load_wye_ids, load_cone_ids)
        var_scen["pd"][i] = JuMP.@variable(
            model,
            [c in load_connections[i]],
            base_name="0_pd_$(i)_$(scen)"
        )
        var_scen["qd"][i] = JuMP.@variable(
            model,
            [c in load_connections[i]],
            base_name="0_qd_$(i)_$(scen)"
        )
    end

    bound = Dict{eltype(load_del_ids), Matrix{Real}}()
    for id in load_del_ids
        load = ref[:load][id]
        bus_id = load["load_bus"]
        bus = ref[:bus][bus_id]
        load_scen = deepcopy(load)
        load_scen["pd"] = load["pd"]*load_factor_scen[id]
        load_scen["qd"] = load["qd"]*load_factor_scen[id]
        cmax = PMD._calc_load_current_max(load_scen, bus)
        bound[id] = bus["vmax"][[findfirst(isequal(c), bus["terminals"]) for c in load_connections[id]]]*cmax'
    end

    cmin = Dict{eltype(load_del_ids), Vector{Real}}()
    cmax = Dict{eltype(load_del_ids), Vector{Real}}()
    for id in load_del_ids
        bus_id = load["load_bus"]
        bus = ref[:bus][bus_id]
        load_scen = deepcopy(load)
        load_scen["pd"] = load["pd"]*load_factor_scen[id]
        load_scen["qd"] = load["qd"]*load_factor_scen[id]
        cmin[id], cmax[id] = PMD._calc_load_current_magnitude_bounds(load_scen, bus)
    end
    (var_scen["Xdr"],var_scen["Xdi"]) = PMD.variable_mx_complex(model, load_del_ids, load_connections, load_connections; symm_bound=bound, name="0_Xd_$(scen)")
    (var_scen["CCdr"], var_scen["CCdi"]) = PMD.variable_mx_hermitian(model, load_del_ids, load_connections; sqrt_upper_bound=cmax, sqrt_lower_bound=cmin, name="0_CCd_$(scen)")

    # variable_mc_capacitor_switch_state
    var_scen["z_cap"] = Dict(
        i => JuMP.@variable(
            model,
            [p in cap["connections"]],
            base_name="0_cap_sw_$(i)_$(scen)",
            binary = true,
        ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
    )

    # variable_mc_capacitor_reactive_power
    var_scen["qc"] = Dict(
        i => JuMP.@variable(
            model,
            [p in cap["connections"]],
            base_name="0_cap_cur_$(i)_$(scen)",
        ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
    )

    # variable representing if switch ab has 'color' k
    if !feas_chck
        var_scen["y"] = Dict()
        for k in L
            for ab in keys(ref[:switch])
                var_scen["y"][(k,ab)] = JuMP.@variable(
                    model,
                    base_name="0_y_gfm[$k,$ab]_$(scen)",
                    binary=true,
                    lower_bound=0,
                    upper_bound=1
                )
            end
        end

        # Eqs. (9)-(10)
        var_scen["f"] = Dict()
        var_scen["ϕ"] = Dict()
        for kk in L # color
            for ab in keys(ref[:switch])
                var_scen["f"][(kk,ab)] = JuMP.@variable(
                    model,
                    base_name="0_f_gfm[$kk,$ab]_$(scen)"
                )
            end
            touched = Set()
            ab = 1
            for k in sort(collect(L)) # fr block
                for k′ in sort(collect(filter(x->x!=k,L))) # to block
                    if (k,k′) ∉ touched
                        push!(touched, (k,k′), (k′,k))
                        var_scen["ϕ"][(kk,ab)] = JuMP.@variable(
                            model,
                            base_name="0_phi_gfm[$kk,$ab]_$(scen)",
                            lower_bound=0,
                            upper_bound=1
                        )
                        ab += 1
                    end
                end
            end
        end
    end

    # power balance constraints
    var_scen["pd_zblock"] = Dict(i => Dict(l => JuMP.@variable(model, [c in conns], base_name="0_pd_zblock_$(l)_$(scen)") for (l,conns) in ref[:bus_conns_load][i]) for (i,bus) in ref[:bus])
    var_scen["qd_zblock"] = Dict(i => Dict(l => JuMP.@variable(model, [c in conns], base_name="0_qd_zblock_$(l)_$(scen)") for (l,conns) in ref[:bus_conns_load][i]) for (i,bus) in ref[:bus])

    # storage constraints
    var_scen["qsc_zblock"] = Dict(i => JuMP.@variable(model, base_name="0_qd_zblock_$(i)_$(scen)") for (i,strg) in ref[:storage] if JuMP.has_lower_bound(var_scen["qsc"][i]) && JuMP.has_upper_bound(var_scen["qsc"][i]))
    var_scen["ps_sqr"] = Dict(i => [JuMP.@variable(model, base_name="0_ps_sqr_$(i)_$(c)_$(scen)") for c in strg["connections"]] for (i,strg) in ref[:storage])
    var_scen["qs_sqr"] = Dict(i => [JuMP.@variable(model, base_name="0_qs_sqr_$(i)_$(c)_$(scen)") for c in strg["connections"]] for (i,strg) in ref[:storage])
    var_scen["sd_on_ps"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_ps_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sc_on_ps"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_ps_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sd_on_qs"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_qs_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sc_on_qs"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_qs_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["ps_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_ps_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["qs_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_qs_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sd_on_ps_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_ps_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sc_on_ps_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_ps_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sd_on_qs_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_qs_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))
    var_scen["sc_on_qs_zinverter"] = Dict(i => JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_qs_zinverter_$(i)_$(scen)") for (i,strg) in ref[:storage] if isfinite(get(strg, "phase_unbalance_factor", Inf)))

    # branch constraints
    var_scen["p_sqr_fr"] = Dict(i => [JuMP.@variable(model, base_name="0_p_sqr_fr_$((i, branch["f_bus"], branch["t_bus"]))[$(c)]_$(scen)") for c in branch["f_connections"]] for (i,branch) in ref[:branch] if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf))
    var_scen["q_sqr_fr"] = Dict(i => [JuMP.@variable(model, base_name="0_q_sqr_fr_$((i, branch["f_bus"], branch["t_bus"]))[$(c)]_$(scen)") for c in branch["f_connections"]] for (i,branch) in ref[:branch] if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf))
    var_scen["p_sqr_to"] = Dict(i => [JuMP.@variable(model, base_name="0_p_sqr_to_$((i, branch["t_bus"], branch["f_bus"]))[$(c)]_$(scen)") for c in branch["t_connections"]] for (i,branch) in ref[:branch] if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf))
    var_scen["q_sqr_to"] = Dict(i => [JuMP.@variable(model, base_name="0_q_sqr_to_$((i, branch["t_bus"], branch["f_bus"]))[$(c)]_$(scen)") for c in branch["t_connections"]] for (i,branch) in ref[:branch] if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf))

    if !feas_chck
        # constraint_switch_close_action_limit
        var_scen["Δᵞs"] = Dict(l => JuMP.@variable(model, base_name="0_delta_switch_state_$(l)_$(scen)") for l in keys(ref[:switch_dispatchable]) if switch_close_actions_ub < Inf)

        # constraint_radial_topology
        var_scen["f_rad"] = Dict()
        var_scen["λ"] = Dict()
        var_scen["β"] = Dict()

        for (i,j) in _L′
            for k in filter(kk->kk∉iᵣ,_N)
                var_scen["f_rad"][(k, i, j)] = JuMP.@variable(model, base_name="0_f_$((k,i,j))_$(scen)")
            end
            var_scen["λ"][(i,j)] = JuMP.@variable(model, base_name="0_lambda_$((i,j))_$(scen)", binary=true, lower_bound=0, upper_bound=1)

            if (i,j) ∈ _L₀
                var_scen["β"][(i,j)] = JuMP.@variable(model, base_name="0_beta_$((i,j))_$(scen)", lower_bound=0, upper_bound=1)
            end
        end
    end

    # switch constraints
    var_scen["psw_sqr_fr"] = Dict(i => [JuMP.@variable(model, base_name="0_psw_sqr_fr_$((i, switch["f_bus"], switch["t_bus"]))[$(c)]_$(scen)") for c in switch["f_connections"]] for (i,switch) in ref[:switch] if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf))
    var_scen["qsw_sqr_fr"] = Dict(i => [JuMP.@variable(model, base_name="0_qsw_sqr_fr_$((i, switch["f_bus"], switch["t_bus"]))[$(c)]_$(scen)") for c in switch["f_connections"]] for (i,switch) in ref[:switch] if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf))
    Dict(i => [JuMP.@variable(model, base_name="0_psw_sqr_fr_$((i, switch["f_bus"], switch["t_bus"]))[$(c)]_$(scen)") for c in switch["f_connections"]] for (i,switch) in ref[:switch] if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf))

    # transformer constraints
    var_scen["tm"] = Dict(trans_id =>
        [transformer["tm_fix"][idx] ? transformer["tm_set"][idx] : var_scen["tap"][trans_id][idx]
        for (idx,(fc,tc)) in enumerate(zip(transformer["f_connections"],transformer["t_connections"]))]
        for (trans_id,transformer) in ref[:transformer] if transformer["configuration"] == PMD.WYE
    )
    var_scen["tmsqr"] = Dict(trans_id => [
        transformer["tm_fix"][i] ? var_scen["tm"][trans_id][i]^2 : JuMP.@variable(
            model,
            base_name="0_tmsqr_$(trans_id)_$(transformer["f_connections"][i])_$(scen)",
            start=JuMP.start_value(var_scen["tm"][trans_id][i])^2,
            lower_bound=JuMP.has_lower_bound(var_scen["tm"][trans_id][i]) ? JuMP.lower_bound(var_scen["tm"][trans_id][i])^2 : 0.9^2,
            upper_bound=JuMP.has_upper_bound(var_scen["tm"][trans_id][i]) ? JuMP.upper_bound(var_scen["tm"][trans_id][i])^2 : 1.1^2
        ) for i in 1:length(var_scen["tm"][trans_id])
    ] for (trans_id,transformer) in ref[:transformer] if transformer["configuration"] == PMD.WYE)
    var_scen["tmsqr_w_to"] = Dict(trans_id =>
        JuMP.@variable(model,
        base_name="0_tmsqr_w_to_$(trans_id)_$(transformer["t_bus"])_$(tc)_$(scen)") for (trans_id,transformer) in ref[:transformer]
        if transformer["configuration"] == PMD.WYE
        for (idx, (fc, tc)) in enumerate(zip(transformer["f_connections"], transformer["t_connections"]))
        if !transformer["tm_fix"][idx]
    )

    # objective
    if !feas_chck
        var_scen["delta_sw_state"] = JuMP.@variable(
            model,
            [i in keys(ref[:switch_dispatchable])],
            base_name="$(i)_delta_sw_state_$(scen)",
        )
    end

end


## add variables common to all scenarios
function variable_common_model(model::JuMP.Model, all_var_common::Dict{Any, Any})

    # variable_inverter_indicator
    all_var_common["z_inverter"] = Dict(
        (t,i) => get(ref[t][i], "inverter", 1) == 1 ? JuMP.@variable(
            model,
            base_name="0_$(t)_z_inverter_$(i)",
            binary=true,
            lower_bound=0,
            upper_bound=1,
        ) : 0 for t in [:storage, :gen] for i in keys(ref[t])
    )

    # variable_switch_state
    all_var_common["z_switch"] = Dict(i => JuMP.@variable(
        model,
        base_name="0_switch_state_$(i)",
        binary=true,
        lower_bound=0,
        upper_bound=1,
    ) for i in keys(ref[:switch_dispatchable]))

end


## add constraints
function constraint_model(model::JuMP.Model, var_scen::Dict{Any, Any}, var_common::Dict{Any, Any} ,load_factor_scen::Dict{Any, Any}; feas_chck::Bool=false)

    # variable_block_indicator
    z_block = var_scen["z_block"]

    # variable_inverter_indicator
    z_inverter = var_common["z_inverter"]

    # variable_mc_bus_voltage_on_off -> variable_mc_bus_voltage_magnitude_sqr_on_off
    w = var_scen["w"]

    # w bounds
    for (i,bus) in ref[:bus]
        for (idx,t) in enumerate(bus["terminals"])
            isfinite(bus["vmax"][idx]) && JuMP.set_upper_bound(w[i][t], bus["vmax"][idx]^2)
        end
    end

    # variable_mc_branch_power
    p = var_scen["p"]
    q = var_scen["q"]

    # p and q bounds
    for (l,i,j) in ref[:arcs_branch]
        smax = PMD._calc_branch_power_max(ref[:branch][l], ref[:bus][i])
        for (idx, c) in enumerate(branch_connections[(l,i,j)])
            PMD.set_upper_bound(p[(l,i,j)][c],  smax[idx])
            PMD.set_lower_bound(p[(l,i,j)][c], -smax[idx])

            PMD.set_upper_bound(q[(l,i,j)][c],  smax[idx])
            PMD.set_lower_bound(q[(l,i,j)][c], -smax[idx])
        end
    end

    # variable_mc_switch_power
    psw = var_scen["psw"]
    qsw = var_scen["qsw"]

    # psw and qsw bounds
    for (l,i,j) in ref[:arcs_switch]
        smax = PMD._calc_branch_power_max(ref[:switch][l], ref[:bus][i])
        for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
            PMD.set_upper_bound(psw[(l,i,j)][c],  smax[idx])
            PMD.set_lower_bound(psw[(l,i,j)][c], -smax[idx])

            PMD.set_upper_bound(qsw[(l,i,j)][c],  smax[idx])
            PMD.set_lower_bound(qsw[(l,i,j)][c], -smax[idx])
        end
    end

    # this explicit type erasure is necessary
    psw_expr = var_scen["psw_expr"]
    qsw_expr = var_scen["qsw_expr"]
    psw_auxes = var_scen["psw_auxes"]
    qsw_auxes = var_scen["qsw_auxes"]

    # This is needed to get around error: "unexpected affine expression in nlconstraint" and overwrite psw/qsw
    for ((l,i,j), psw_aux) in psw_auxes
        for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
            JuMP.@constraint(model, psw_expr[(l,i,j)][c] == psw_aux[c])
        end
    end
    for (k,psw_aux) in psw_auxes
        psw[k] = psw_aux
    end

    for ((l,i,j), qsw_aux) in qsw_auxes
        for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
            JuMP.@constraint(model, qsw_expr[(l,i,j)][c] == qsw_aux[c])
        end
    end
    for (k,qsw_aux) in qsw_auxes
        qsw[k] = qsw_aux
    end

    # variable_switch_state
    z_switch = var_common["z_switch"]

    # fixed switches
    for i in [i for i in keys(ref[:switch]) if !(i in keys(ref[:switch_dispatchable]))]
        z_switch[i] = ref[:switch][i]["state"]
    end

    # variable_mc_transformer_power
    pt = var_scen["pt"]
    qt = var_scen["qt"]

    # pt and qt bounds
    for arc in ref[:arcs_transformer_from]
        (l,i,j) = arc
        rate_a_fr, rate_a_to = PMD._calc_transformer_power_ub_frto(ref[:transformer][l], ref[:bus][i], ref[:bus][j])

        for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)]))
            PMD.set_lower_bound(pt[(l,i,j)][fc], -rate_a_fr[idx])
            PMD.set_upper_bound(pt[(l,i,j)][fc],  rate_a_fr[idx])
            PMD.set_lower_bound(pt[(l,j,i)][tc], -rate_a_to[idx])
            PMD.set_upper_bound(pt[(l,j,i)][tc],  rate_a_to[idx])

            PMD.set_lower_bound(qt[(l,i,j)][fc], -rate_a_fr[idx])
            PMD.set_upper_bound(qt[(l,i,j)][fc],  rate_a_fr[idx])
            PMD.set_lower_bound(qt[(l,j,i)][tc], -rate_a_to[idx])
            PMD.set_upper_bound(qt[(l,j,i)][tc],  rate_a_to[idx])
        end
    end

    # variable_mc_oltc_transformer_tap
    tap = var_scen["tap"]

    # tap bounds
    for tr_id in p_oltc_ids, p in 1:length(ref[:transformer][tr_id]["f_connections"])
        PMD.set_lower_bound(tap[tr_id][p], ref[:transformer][tr_id]["tm_lb"][p])
        PMD.set_upper_bound(tap[tr_id][p], ref[:transformer][tr_id]["tm_ub"][p])
    end

    # variable_mc_generator_power_on_off
    pg = var_scen["pg"]
    qg = var_scen["qg"]

    # pg and qg bounds
    for (i,gen) in ref[:gen]
        for (idx,c) in enumerate(gen["connections"])
            isfinite(gen["pmin"][idx]) && JuMP.set_lower_bound(pg[i][c], min(0.0, gen["pmin"][idx]))
            isfinite(gen["pmax"][idx]) && JuMP.set_upper_bound(pg[i][c], gen["pmax"][idx])

            isfinite(gen["qmin"][idx]) && JuMP.set_lower_bound(qg[i][c], min(0.0, gen["qmin"][idx]))
            isfinite(gen["qmax"][idx]) && JuMP.set_upper_bound(qg[i][c], gen["qmax"][idx])
        end
    end

    # variable_mc_storage_power_on_off and variable_mc_storage_power_control_imaginary_on_off
    ps = var_scen["ps"]
    qs = var_scen["qs"]
    qsc = var_scen["qsc"]

    # ps, qs bounds
    for (i,strg) in ref[:storage]
            for (idx, c) in enumerate(strg["connections"])
            if !isinf(storage_inj_lb[i][idx])
                PMD.set_lower_bound(ps[i][c], storage_inj_lb[i][idx])
                PMD.set_lower_bound(qs[i][c], storage_inj_lb[i][idx])
            end
            if !isinf(storage_inj_ub[i][idx])
                PMD.set_upper_bound(ps[i][c], storage_inj_ub[i][idx])
                PMD.set_upper_bound(qs[i][c], storage_inj_ub[i][idx])
            end
        end
    end

    # variable_storage_energy, variable_storage_charge and variable_storage_discharge
    se = var_scen["se"]
    sc = var_scen["sc"]
    sd = var_scen["sd"]

    # se, sc and sd bounds
    for (i, storage) in ref[:storage]
        PMD.set_upper_bound(se[i], storage["energy_rating"])
        PMD.set_upper_bound(sc[i], storage["charge_rating"])
        PMD.set_upper_bound(sd[i], storage["discharge_rating"])
    end

    # variable_storage_complementary_indicator and variable_storage_complementary_indicator
    sc_on = var_scen["sc_on"]
    sd_on = var_scen["sd_on"]

    # load variables
    pd = var_scen["pd"]
    qd = var_scen["qd"]
    pd_bus = var_scen["pd_bus"]
    qd_bus = var_scen["qd_bus"]
    Xdr = var_scen["Xdr"]
    Xdi = var_scen["Xdi"]
    CCr = var_scen["CCdr"]
    CCi = var_scen["CCdi"]
    for i in intersect(load_wye_ids, load_cone_ids)
        load = ref[:load][i]
        load_scen = deepcopy(load)
        load_scen["pd"] = load["pd"]*load_factor_scen[i]
        load_scen["qd"] = load["qd"]*load_factor_scen[i]
        bus = ref[:bus][load["load_bus"]]
        pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load_scen, bus)

        for (idx,c) in enumerate(load_connections[i])
            PMD.set_lower_bound(pd[i][c], pmin[idx])
            PMD.set_upper_bound(pd[i][c], pmax[idx])
            PMD.set_lower_bound(qd[i][c], qmin[idx])
            PMD.set_upper_bound(qd[i][c], qmax[idx])
        end
    end

    # variable_mc_capacitor_switch_state
    z_cap = var_scen["z_cap"]

    # variable_mc_capacitor_reactive_power
    qc = var_scen["qc"]

    # voltage sources are always grid-forming
    for ((t,j), z_inv) in z_inverter
        if t == :gen && startswith(ref[t][j]["source_id"], "voltage_source")
            JuMP.@constraint(model, z_inv == z_block[ref[:bus_block_map][ref[t][j]["$(t)_bus"]]])
        end
    end

    if !feas_chck
        # variable representing if switch ab has 'color' k
        y = var_scen["y"]

        # Eqs. (9)-(10)
        f = var_scen["f"]
        ϕ = var_scen["ϕ"]
        for kk in L # color
            for ab in keys(ref[:switch])
                JuMP.@constraint(model, f[kk,ab] >= -length(keys(ref[:switch]))*(z_switch[ab]))
                JuMP.@constraint(model, f[kk,ab] <=  length(keys(ref[:switch]))*(z_switch[ab]))
            end
        end

        # constrain each y to have only one color
        for ab in keys(ref[:switch])
            JuMP.@constraint(model, sum(y[(k,ab)] for k in L) <= z_switch[ab])
        end
    end

    # Eqs. (3)-(7)
    for k in L
        Dₖ = ref[:block_inverters][k]
        Tₖ = ref[:block_switches][k]

        if !isempty(Dₖ)
            # Eq. (14)
            JuMP.@constraint(model, sum(z_inverter[i] for i in Dₖ) >= sum(1-z_switch[ab] for ab in Tₖ)-length(Tₖ)+z_block[k])
            JuMP.@constraint(model, sum(z_inverter[i] for i in Dₖ) <= z_block[k])

            if !feas_chck
                # Eq. (4)-(5)
                for (t,j) in Dₖ
                    if t == :storage
                        pmin = fill(-Inf, length(ref[t][j]["connections"]))
                        pmax = fill( Inf, length(ref[t][j]["connections"]))
                        qmin = fill(-Inf, length(ref[t][j]["connections"]))
                        qmax = fill( Inf, length(ref[t][j]["connections"]))

                        for (idx,c) in enumerate(ref[t][j]["connections"])
                            pmin[idx] = storage_inj_lb[j][idx]
                            pmax[idx] = storage_inj_ub[j][idx]
                            qmin[idx] = max(storage_inj_lb[j][idx], ref[t][j]["qmin"])
                            qmax[idx] = min(storage_inj_ub[j][idx], ref[t][j]["qmax"])

                            if isfinite(pmax[idx]) && pmax[idx] >= 0
                                JuMP.@constraint(model, ps[j][c] <= pmax[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, ps[j][c] <= pmax[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(qmax[idx]) && qmax[idx] >= 0
                                JuMP.@constraint(model, qs[j][c] <= qmax[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, qs[j][c] <= qmax[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(pmin[idx]) && pmin[idx] <= 0
                                JuMP.@constraint(model, ps[j][c] >= pmin[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, ps[j][c] >= pmin[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(qmin[idx]) && qmin[idx] <= 0
                                JuMP.@constraint(model, qs[j][c] >= qmin[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, qs[j][c] >= qmin[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                        end
                    elseif t == :gen
                        pmin = ref[t][j]["pmin"]
                        pmax = ref[t][j]["pmax"]
                        qmin = ref[t][j]["qmin"]
                        qmax = ref[t][j]["qmax"]

                        for (idx,c) in enumerate(ref[t][j]["connections"])
                            if isfinite(pmax[idx]) && pmax[idx] >= 0
                                JuMP.@constraint(model, pg[j][c] <= pmax[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, pg[j][c] <= pmax[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(qmax[idx]) && qmax[idx] >= 0
                                JuMP.@constraint(model, qg[j][c] <= qmax[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, qg[j][c] <= qmax[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(pmin[idx]) && pmin[idx] <= 0
                                JuMP.@constraint(model, pg[j][c] >= pmin[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, pg[j][c] >= pmin[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                            if isfinite(qmin[idx]) && qmin[idx] <= 0
                                JuMP.@constraint(model, qg[j][c] >= qmin[idx] * (sum(z_switch[ab] for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                                JuMP.@constraint(model, qg[j][c] >= qmin[idx] * (sum(y[(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[i] for i in Dₖ)))
                            end
                        end
                    end
                end
            end
        end

        if !feas_chck
            for ab in Tₖ
                # Eq. (6)
                JuMP.@constraint(model, sum(z_inverter[i] for i in Dₖ) >= y[(k, ab)] - (1 - z_switch[ab]))
                JuMP.@constraint(model, sum(z_inverter[i] for i in Dₖ) <= y[(k, ab)] + (1 - z_switch[ab]))

                # Eq. (8)
                JuMP.@constraint(model, y[(k,ab)] <= sum(z_inverter[i] for i in Dₖ))

                for dc in filter(x->x!=ab, Tₖ)
                    for k′ in L
                        # Eq. (7)
                        JuMP.@constraint(model, y[(k′,ab)] >= y[(k′,dc)] - (1 - z_switch[dc]) - (1 - z_switch[ab]))
                        JuMP.@constraint(model, y[(k′,ab)] <= y[(k′,dc)] + (1 - z_switch[dc]) + (1 - z_switch[ab]))
                    end
                end
            end

            # Eq. (11)
            JuMP.@constraint(model, sum(f[(k,ab)] for ab in filter(x->map_id_pairs[x][1] == k, Tₖ)) - sum(f[(k,ab)] for ab in filter(x->map_id_pairs[x][2] == k, Tₖ)) + sum(ϕ[(k,ab)] for ab in Φₖ[k]) == length(L) - 1)

            # Eq. (15)
            JuMP.@constraint(model, z_block[k] <= sum(z_inverter[i] for i in Dₖ) + sum(y[(k′,ab)] for k′ in L for ab in Tₖ))

            for k′ in filter(x->x!=k, L)
                Tₖ′ = ref[:block_switches][k′]
                kk′ = map_virtual_pairs_id[k][(k,k′)]

                # Eq. (12)
                JuMP.@constraint(model, sum(f[(k,ab)] for ab in filter(x->map_id_pairs[x][1]==k′, Tₖ′)) - sum(f[(k,ab)] for ab in filter(x->map_id_pairs[x][2]==k′, Tₖ′)) - ϕ[(k,(kk′))] == -1)

                # Eq. (13)
                for ab in Tₖ′
                    JuMP.@constraint(model, y[k,ab] <= 1 - ϕ[(k,kk′)])
                end
            end
        end
    end

    # constraint_mc_inverter_theta_ref
    for (i,bus) in ref[:bus]
        # reference bus "theta" constraint
        vmax = min(bus["vmax"]..., 2.0)
        if isfinite(vmax)
            if length(w[i]) > 1 && !isempty([z_inverter[inv_obj] for inv_obj in ref[:bus_inverters][i]])
                for t in 2:length(w[i])
                    JuMP.@constraint(model, w[i][t] - w[i][1] <=  vmax^2 * (1 - sum([z_inverter[inv_obj] for inv_obj in ref[:bus_inverters][i]])))
                    JuMP.@constraint(model, w[i][t] - w[i][1] >= -vmax^2 * (1 - sum([z_inverter[inv_obj] for inv_obj in ref[:bus_inverters][i]])))
                end
            end
        end
    end

    # constraint_mc_bus_voltage_block_on_off
    for (i,bus) in ref[:bus]
        # bus voltage on off constraint
        for (idx,t) in [(idx,t) for (idx,t) in enumerate(bus["terminals"]) if !bus["grounded"][idx]]
            isfinite(bus["vmax"][idx]) && JuMP.@constraint(model, w[i][t] <= bus["vmax"][idx]^2*z_block[ref[:bus_block_map][i]])
            isfinite(bus["vmin"][idx]) && JuMP.@constraint(model, w[i][t] >= bus["vmin"][idx]^2*z_block[ref[:bus_block_map][i]])
        end
    end

    # constraint_mc_generator_power_block_on_off
    for (i,gen) in ref[:gen]
        for (idx, c) in enumerate(gen["connections"])
            isfinite(gen["pmin"][idx]) && JuMP.@constraint(model, pg[i][c] >= gen["pmin"][idx]*z_block[ref[:gen_block_map][i]])
            isfinite(gen["qmin"][idx]) && JuMP.@constraint(model, qg[i][c] >= gen["qmin"][idx]*z_block[ref[:gen_block_map][i]])

            isfinite(gen["pmax"][idx]) && JuMP.@constraint(model, pg[i][c] <= gen["pmax"][idx]*z_block[ref[:gen_block_map][i]])
            isfinite(gen["qmax"][idx]) && JuMP.@constraint(model, qg[i][c] <= gen["qmax"][idx]*z_block[ref[:gen_block_map][i]])
        end
    end

    # constraint_mc_load_power
    for (load_id,load) in ref[:load]
        bus_id = load["load_bus"]
        bus = ref[:bus][bus_id]
        Td = [1 -1 0; 0 1 -1; -1 0 1]
        load_scen = deepcopy(load)
        load_scen["pd"] = load["pd"]*load_factor_scen[load_id]
        load_scen["qd"] = load["qd"]*load_factor_scen[load_id]
        a, alpha, b, beta = PMD._load_expmodel_params(load_scen, bus)
        pd0 = load_scen["pd"]
        qd0 = load_scen["qd"]
        if load["configuration"]==PMD.WYE
            if load["model"]==PMD.POWER
                pd[load_id] = JuMP.Containers.DenseAxisArray(pd0, load["connections"])
                qd[load_id] = JuMP.Containers.DenseAxisArray(qd0, load["connections"])
            elseif load["model"]==PMD.IMPEDANCE
                _w = w[bus_id][[c for c in load["connections"]]]
                pd[load_id] = a.*_w
                qd[load_id] = b.*_w
            else
                for (idx,c) in enumerate(load["connections"])
                    JuMP.@constraint(model, pd[load_id][c]==1/2*a[idx]*(w[bus_id][c]+1))
                    JuMP.@constraint(model, qd[load_id][c]==1/2*b[idx]*(w[bus_id][c]+1))
                end
            end

            pd_bus[load_id] = pd[load_id]
            qd_bus[load_id] = qd[load_id]

        elseif load["configuration"]==PMD.DELTA
            pd_bus[load_id] = LinearAlgebra.diag(Xdr[load_id]*Td)
            qd_bus[load_id] = LinearAlgebra.diag(Xdi[load_id]*Td)
            pd[load_id] = LinearAlgebra.diag(Td*Xdr[load_id])
            qd[load_id] = LinearAlgebra.diag(Td*Xdi[load_id])

            for (idx, c) in enumerate(load["connections"])
                if abs(pd0[idx]+im*qd0[idx]) == 0.0
                    JuMP.@constraint(model, Xdr[load_id][:,idx] .== 0)
                    JuMP.@constraint(model, Xdi[load_id][:,idx] .== 0)
                end
            end

            if load["model"]==PMD.POWER
                for (idx, c) in enumerate(load["connections"])
                    JuMP.@constraint(model, pd[load_id][idx]==pd0[idx])
                    JuMP.@constraint(model, qd[load_id][idx]==qd0[idx])
                end
            elseif load["model"]==PMD.IMPEDANCE
                for (idx,c) in enumerate(load["connections"])
                    JuMP.@constraint(model, pd[load_id][idx]==3*a[idx]*w[bus_id][[c for c in load["connections"]]][idx])
                    JuMP.@constraint(model, qd[load_id][idx]==3*b[idx]*w[bus_id][[c for c in load["connections"]]][idx])
                end
            else
                for (idx,c) in enumerate(load["connections"])
                    JuMP.@constraint(model, pd[load_id][idx]==sqrt(3)/2*a[idx]*(w[bus_id][[c for c in load["connections"]]][idx]+1))
                    JuMP.@constraint(model, qd[load_id][idx]==sqrt(3)/2*b[idx]*(w[bus_id][[c for c in load["connections"]]][idx]+1))
                end
            end
        end
    end

    # power balance constraints
    for (i,bus) in ref[:bus]
        uncontrolled_shunts = Tuple{Int,Vector{Int}}[]
        controlled_shunts = Tuple{Int,Vector{Int}}[]

        if !isempty(ref[:bus_conns_shunt][i]) && any(haskey(ref[:shunt][sh], "controls") for (sh, conns) in ref[:bus_conns_shunt][i])
            for (sh, conns) in ref[:bus_conns_shunt][i]
                if haskey(ref[:shunt][sh], "controls")
                    push!(controlled_shunts, (sh,conns))
                else
                    push!(uncontrolled_shunts, (sh, conns))
                end
            end
        else
            uncontrolled_shunts = ref[:bus_conns_shunt][i]
        end

        Gt, _ = build_bus_shunt_matrices(ref, bus["terminals"], ref[:bus_conns_shunt][i])
        _, Bt = build_bus_shunt_matrices(ref, bus["terminals"], uncontrolled_shunts)

        ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(bus["terminals"]) if !bus["grounded"][idx]]

        pd_zblock = var_scen["pd_zblock"][i]
        qd_zblock = var_scen["qd_zblock"][i]

        for (l,conns) in ref[:bus_conns_load][i]
            for c in conns
                IM.relaxation_product(model, pd_bus[l][c], z_block[ref[:load_block_map][l]], pd_zblock[l][c])
                IM.relaxation_product(model, qd_bus[l][c], z_block[ref[:load_block_map][l]], qd_zblock[l][c])
            end
        end

        for (idx, t) in ungrounded_terminals
            JuMP.@constraint(model,
                sum(p[a][t] for (a, conns) in ref[:bus_arcs_conns_branch][i] if t in conns)
                + sum(psw[a_sw][t] for (a_sw, conns) in ref[:bus_arcs_conns_switch][i] if t in conns)
                + sum(pt[a_trans][t] for (a_trans, conns) in ref[:bus_arcs_conns_transformer][i] if t in conns)
                ==
                sum(pg[g][t] for (g, conns) in ref[:bus_conns_gen][i] if t in conns)
                - sum(ps[s][t] for (s, conns) in ref[:bus_conns_storage][i] if t in conns)
                - sum(pd_zblock[l][t] for (l, conns) in ref[:bus_conns_load][i] if t in conns)
                - sum((w[i][t] * LinearAlgebra.diag(Gt')[idx]) for (sh, conns) in ref[:bus_conns_shunt][i] if t in conns)
            )

            JuMP.@constraint(model,
                sum(q[a][t] for (a, conns) in ref[:bus_arcs_conns_branch][i] if t in conns)
                + sum(qsw[a_sw][t] for (a_sw, conns) in ref[:bus_arcs_conns_switch][i] if t in conns)
                + sum(qt[a_trans][t] for (a_trans, conns) in ref[:bus_arcs_conns_transformer][i] if t in conns)
                ==
                sum(qg[g][t] for (g, conns) in ref[:bus_conns_gen][i] if t in conns)
                - sum(qs[s][t] for (s, conns) in ref[:bus_conns_storage][i] if t in conns)
                - sum(qd_zblock[l][t] for (l, conns) in ref[:bus_conns_load][i] if t in conns)
                - sum((-w[i][t] * LinearAlgebra.diag(Bt')[idx]) for (sh, conns) in uncontrolled_shunts if t in conns)
                - sum(-qc[sh][t] for (sh, conns) in controlled_shunts if t in conns)
            )

            for (sh, sh_conns) in controlled_shunts
                if t in sh_conns
                    bs = LinearAlgebra.diag(ref[:shunt][sh]["bs"])[findfirst(isequal(t), sh_conns)]

                    w_lb, w_ub = IM.variable_domain(w[i][t])
                    JuMP.@constraint(model, z_cap[sh] <= z_block[ref[:bus_block_map][i]])
                    JuMP.@constraint(model, qc[sh] ≥ bs*z_cap[sh]*w_lb)
                    JuMP.@constraint(model, qc[sh] ≥ bs*w[t] + bs*z_cap[sh]*w_ub - bs*w_ub*z_block[ref[:bus_block_map][i]])
                    JuMP.@constraint(model, qc[sh] ≤ bs*z_cap[sh]*w_ub)
                    JuMP.@constraint(model, qc[sh] ≤ bs*w[t] + bs*z_cap[sh]*w_lb - bs*w_lb*z_block[ref[:bus_block_map][i]])
                end
            end
        end
    end

    # storage constraints
    for (i,strg) in ref[:storage]

        # constraint_storage_state
        JuMP.@constraint(model, se[i] - strg["energy"] == ref[:time_elapsed]*(strg["charge_efficiency"]*sc[i] - sd[i]/strg["discharge_efficiency"]))

        # constraint_storage_complementarity_mi_block_on_off
        JuMP.@constraint(model, sc_on[i] + sd_on[i] == z_block[ref[:storage_block_map][i]])
        JuMP.@constraint(model, sc_on[i]*strg["charge_rating"] >= sc[i])
        JuMP.@constraint(model, sd_on[i]*strg["discharge_rating"] >= sd[i])

        # constraint_mc_storage_block_on_off
        ncnds = length(strg["connections"])
        pmin = zeros(ncnds)
        pmax = zeros(ncnds)
        qmin = zeros(ncnds)
        qmax = zeros(ncnds)

        for (idx,c) in enumerate(strg["connections"])
            pmin[idx] = storage_inj_lb[i][idx]
            pmax[idx] = storage_inj_ub[i][idx]
            qmin[idx] = max(storage_inj_lb[i][idx], strg["qmin"])
            qmax[idx] = min(storage_inj_ub[i][idx], strg["qmax"])
        end

        pmin = maximum(pmin)
        pmax = minimum(pmax)
        qmin = maximum(qmin)
        qmax = minimum(qmax)

        isfinite(pmin) && JuMP.@constraint(model, sum(ps[i]) >= z_block[ref[:storage_block_map][i]]*pmin)
        isfinite(qmin) && JuMP.@constraint(model, sum(qs[i]) >= z_block[ref[:storage_block_map][i]]*qmin)

        isfinite(pmax) && JuMP.@constraint(model, sum(ps[i]) <= z_block[ref[:storage_block_map][i]]*pmax)
        isfinite(qmax) && JuMP.@constraint(model, sum(qs[i]) <= z_block[ref[:storage_block_map][i]]*qmax)

        # constraint_mc_storage_losses_block_on_off
        if JuMP.has_lower_bound(qsc[i]) && JuMP.has_upper_bound(qsc[i])
            qsc_zblock = var_scen["qsc_zblock"][i]

            JuMP.@constraint(model, qsc_zblock >= JuMP.lower_bound(qsc[i]) * z_block[ref[:storage_block_map][i]])
            JuMP.@constraint(model, qsc_zblock >= JuMP.upper_bound(qsc[i]) * z_block[ref[:storage_block_map][i]] + qsc[i] - JuMP.upper_bound(qsc[i]))
            JuMP.@constraint(model, qsc_zblock <= JuMP.upper_bound(qsc[i]) * z_block[ref[:storage_block_map][i]])
            JuMP.@constraint(model, qsc_zblock <= qsc[i] + JuMP.lower_bound(qsc[i]) * z_block[ref[:storage_block_map][i]] - JuMP.lower_bound(qsc[i]))

            JuMP.@constraint(model, sum(qs[i]) == qsc_zblock + strg["q_loss"] * z_block[ref[:storage_block_map][i]])
        else
            # Note that this is not supported in LP solvers when z_block is continuous
            JuMP.@constraint(model, sum(qs[i]) == qsc[i] * z_block[ref[:storage_block_map][i]] + strg["q_loss"] * z_block[ref[:storage_block_map][i]])
        end
        JuMP.@constraint(model, sum(ps[i]) + (sd[i] - sc[i]) == strg["p_loss"] * z_block[ref[:storage_block_map][i]])

        # constraint_mc_storage_thermal_limit
        _ps = [ps[i][c] for c in strg["connections"]]
        _qs = [qs[i][c] for c in strg["connections"]]

        ps_sqr = var_scen["ps_sqr"][i]
        qs_sqr = var_scen["qs_sqr"][i]

        for (idx,c) in enumerate(strg["connections"])
            ps_lb, ps_ub = IM.variable_domain(_ps[idx])
            PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, _ps[idx], ps_sqr[idx], [ps_lb, ps_ub], false)

            qs_lb, qs_ub = IM.variable_domain(_qs[idx])
            PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, _qs[idx], qs_sqr[idx], [qs_lb, qs_ub], false)
        end

        JuMP.@constraint(model, sum(ps_sqr .+ qs_sqr) <= strg["thermal_rating"]^2)

        # constraint_mc_storage_phase_unbalance_grid_following
        unbalance_factor = get(strg, "phase_unbalance_factor", Inf)
        if isfinite(unbalance_factor)
            sd_on_ps = var_scen["sd_on_ps"][i]
            sc_on_ps = var_scen["sc_on_ps"][i]
            sd_on_qs = var_scen["sd_on_qs"][i]
            sc_on_qs = var_scen["sc_on_qs"][i]
            ps_zinverter = var_scen["ps_zinverter"][i]
            qs_zinverter = var_scen["qs_zinverter"][i]
            sd_on_ps_zinverter = var_scen["sd_on_ps_zinverter"][i]
            sc_on_ps_zinverter = var_scen["sc_on_ps_zinverter"][i]
            sd_on_qs_zinverter = var_scen["sd_on_qs_zinverter"][i]
            sc_on_qs_zinverter = var_scen["sc_on_qs_zinverter"][i]
            for c in strg["connections"]
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sd_on[i], ps[i][c], sd_on_ps[c], [0,1], [JuMP.lower_bound(ps[i][c]), JuMP.upper_bound(ps[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sc_on[i], ps[i][c], sc_on_ps[c], [0,1], [JuMP.lower_bound(ps[i][c]), JuMP.upper_bound(ps[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sd_on[i], qs[i][c], sd_on_qs[c], [0,1], [JuMP.lower_bound(qs[i][c]), JuMP.upper_bound(qs[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sc_on[i], qs[i][c], sc_on_qs[c], [0,1], [JuMP.lower_bound(qs[i][c]), JuMP.upper_bound(qs[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], ps[i][c], ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[i][c]), JuMP.upper_bound(ps[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], qs[i][c], qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[i][c]), JuMP.upper_bound(qs[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], sd_on_ps[c], sd_on_ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[i][c]), JuMP.upper_bound(ps[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], sc_on_ps[c], sc_on_ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[i][c]), JuMP.upper_bound(ps[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], sd_on_qs[c], sd_on_qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[i][c]), JuMP.upper_bound(qs[i][c])])
                PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[(:storage,i)], sc_on_qs[c], sc_on_qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[i][c]), JuMP.upper_bound(qs[i][c])])
            end

            for (idx,c) in enumerate(strg["connections"])
                if idx < length(strg["connections"])
                    for d in strg["connections"][idx+1:end]
                        JuMP.@constraint(model, ps[i][c]-ps_zinverter[c] >= ps[i][d] - unbalance_factor*(-1*sd_on_ps[d] + 1*sc_on_ps[d]) - ps_zinverter[d] + unbalance_factor*(-1*sd_on_ps_zinverter[d] + 1*sc_on_ps_zinverter[d]))
                        JuMP.@constraint(model, ps[i][c]-ps_zinverter[c] <= ps[i][d] + unbalance_factor*(-1*sd_on_ps[d] + 1*sc_on_ps[d]) - ps_zinverter[d] - unbalance_factor*(-1*sd_on_ps_zinverter[d] + 1*sc_on_ps_zinverter[d]))

                        JuMP.@constraint(model, qs[i][c]-qs_zinverter[c] >= qs[i][d] - unbalance_factor*(-1*sd_on_qs[d] + 1*sc_on_qs[d]) - qs_zinverter[d] + unbalance_factor*(-1*sd_on_qs_zinverter[d] + 1*sc_on_qs_zinverter[d]))
                        JuMP.@constraint(model, qs[i][c]-qs_zinverter[c] <= qs[i][d] + unbalance_factor*(-1*sd_on_qs[d] + 1*sc_on_qs[d]) - qs_zinverter[d] - unbalance_factor*(-1*sd_on_qs_zinverter[d] + 1*sc_on_qs_zinverter[d]))
                    end
                end
            end
        end
    end

    # branch constraints
    for (i,branch) in ref[:branch]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        f_idx = (i, f_bus, t_bus)
        t_idx = (i, t_bus, f_bus)

        r = branch["br_r"]
        x = branch["br_x"]
        g_sh_fr = branch["g_fr"]
        g_sh_to = branch["g_to"]
        b_sh_fr = branch["b_fr"]
        b_sh_to = branch["b_to"]

        f_connections = branch["f_connections"]
        t_connections = branch["t_connections"]
        N = length(f_connections)

        alpha = exp(-im*2*pi/3)
        Gamma = [1 alpha^2 alpha; alpha 1 alpha^2; alpha^2 alpha 1][f_connections,t_connections]
        MP = 2*(real(Gamma).*r + imag(Gamma).*x)
        MQ = 2*(real(Gamma).*x - imag(Gamma).*r)

        p_fr = p[f_idx]
        q_fr = q[f_idx]

        p_to = p[t_idx]
        q_to = q[t_idx]

        w_fr = w[f_bus]
        w_to = w[t_bus]

        # constraint_mc_power_losses
        for (idx, (fc,tc)) in enumerate(zip(f_connections, t_connections))
            JuMP.@constraint(model, p_fr[fc] + p_to[tc] == g_sh_fr[idx,idx]*w_fr[fc] +  g_sh_to[idx,idx]*w_to[tc])
            JuMP.@constraint(model, q_fr[fc] + q_to[tc] == -b_sh_fr[idx,idx]*w_fr[fc] + -b_sh_to[idx,idx]*w_to[tc])
        end

        p_s_fr = [p_fr[fc]- LinearAlgebra.diag(g_sh_fr)[idx].*w_fr[fc] for (idx,fc) in enumerate(f_connections)]
        q_s_fr = [q_fr[fc]+ LinearAlgebra.diag(b_sh_fr)[idx].*w_fr[fc] for (idx,fc) in enumerate(f_connections)]

        # constraint_mc_model_voltage_magnitude_difference
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
            JuMP.@constraint(model, w_to[tc] == w_fr[fc] - sum(MP[idx,j]*p_s_fr[j] for j in 1:N) - sum(MQ[idx,j]*q_s_fr[j] for j in 1:N))
        end

        # constraint_mc_voltage_angle_difference
        for (idx, (fc, tc)) in enumerate(zip(branch["f_connections"], branch["t_connections"]))
            g_fr = branch["g_fr"][idx,idx]
            g_to = branch["g_to"][idx,idx]
            b_fr = branch["b_fr"][idx,idx]
            b_to = branch["b_to"][idx,idx]

            r = branch["br_r"][idx,idx]
            x = branch["br_x"][idx,idx]

            angmin = branch["angmin"]
            angmax = branch["angmax"]

            w_fr = w[f_bus][fc]
            p_fr = p[f_idx][fc]
            q_fr = q[f_idx][fc]

            JuMP.@constraint(model,
                tan(angmin[idx])*((1 + r*g_fr - x*b_fr)*(w_fr) - r*p_fr - x*q_fr)
                        <= ((-x*g_fr - r*b_fr)*(w_fr) + x*p_fr - r*q_fr)
                )
            JuMP.@constraint(model,
                tan(angmax[idx])*((1 + r*g_fr - x*b_fr)*(w_fr) - r*p_fr - x*q_fr)
                        >= ((-x*g_fr - r*b_fr)*(w_fr) + x*p_fr - r*q_fr)
                )
        end

        # ampacity constraints
        if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf)
            c_rating = branch["c_rating_a"]

            # constraint_mc_ampacity_from
            p_fr = [p[f_idx][c] for c in f_connections]
            q_fr = [q[f_idx][c] for c in f_connections]
            w_fr = [w[f_idx[2]][c] for c in f_connections]

            p_sqr_fr = var_scen["p_sqr_fr"][i]
            q_sqr_fr = var_scen["q_sqr_fr"][i]

            for (idx,c) in enumerate(f_connections)
                if isfinite(c_rating[idx])
                    p_lb, p_ub = IM.variable_domain(p_fr[idx])
                    q_lb, q_ub = IM.variable_domain(q_fr[idx])
                    w_ub = IM.variable_domain(w_fr[idx])[2]

                    if (!isfinite(p_lb) || !isfinite(p_ub)) && isfinite(w_ub)
                        p_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        p_lb = -p_ub
                    end
                    if (!isfinite(q_lb) || !isfinite(q_ub)) && isfinite(w_ub)
                        q_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        q_lb = -q_ub
                    end

                    all(isfinite(b) for b in [p_lb, p_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, p_fr[idx], p_sqr_fr[idx], [p_lb, p_ub], false)
                    all(isfinite(b) for b in [q_lb, q_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, q_fr[idx], q_sqr_fr[idx], [q_lb, q_ub], false)
                end
            end

            # constraint_mc_ampacity_to
            p_to = [p[t_idx][c] for c in t_connections]
            q_to = [q[t_idx][c] for c in t_connections]
            w_to = [w[t_idx[2]][c] for c in t_connections]

            p_sqr_to = var_scen["p_sqr_to"][i]
            q_sqr_to = var_scen["q_sqr_to"][i]

            for (idx,c) in enumerate(t_connections)
                if isfinite(c_rating[idx])
                    p_lb, p_ub = IM.variable_domain(p_to[idx])
                    q_lb, q_ub = IM.variable_domain(q_to[idx])
                    w_ub = IM.variable_domain(w_to[idx])[2]

                    if (!isfinite(p_lb) || !isfinite(p_ub)) && isfinite(w_ub)
                        p_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        p_lb = -p_ub
                    end
                    if (!isfinite(q_lb) || !isfinite(q_ub)) && isfinite(w_ub)
                        q_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        q_lb = -q_ub
                    end

                    all(isfinite(b) for b in [p_lb, p_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, p_to[idx], p_sqr_to[idx], [p_lb, p_ub], false)
                    all(isfinite(b) for b in [q_lb, q_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, q_to[idx], q_sqr_to[idx], [q_lb, q_ub], false)
                end
            end
        end
    end

    if !feas_chck
        # constraint_switch_close_action_limit
        if switch_close_actions_ub < Inf
            Δᵞs = var_scen["Δᵞs"]

            for (s, Δᵞ) in Δᵞs
                γ = z_switch[s]
                γ₀ = JuMP.start_value(γ)
                JuMP.@constraint(model, Δᵞ >=  γ * (1 - γ₀))
                JuMP.@constraint(model, Δᵞ >= -γ * (1 - γ₀))
            end
            JuMP.@constraint(model, sum(Δᵞ for (l, Δᵞ) in Δᵞs) <= switch_close_actions_ub)
        end

        # constraint_radial_topology
        f_rad = var_scen["f_rad"]
        λ = var_scen["λ"]
        β = var_scen["β"]
        α = Dict()

        for (s,sw) in ref[:switch]
            (i,j) = (ref[:bus_block_map][sw["f_bus"]], ref[:bus_block_map][sw["t_bus"]])
            α[(i,j)] = z_switch[s]
        end

        for (i,j) in _L′
            for k in filter(kk->kk∉iᵣ,_N)
                f_rad[(k, i, j)] = JuMP.@variable(model, base_name="0_f_$((k,i,j))")
            end
            λ[(i,j)] = JuMP.@variable(model, base_name="0_lambda_$((i,j))", binary=true, lower_bound=0, upper_bound=1)

            if (i,j) ∈ _L₀
                β[(i,j)] = JuMP.@variable(model, base_name="0_beta_$((i,j))", lower_bound=0, upper_bound=1)
            end
        end

        JuMP.@constraint(model, sum((λ[(i,j)] + λ[(j,i)]) for (i,j) in _L) == length(_N) - 1)

        for (i,j) in _L₀
            JuMP.@constraint(model, λ[(i,j)] + λ[(j,i)] == β[(i,j)])
            JuMP.@constraint(model, α[(i,j)] <= β[(i,j)])
        end

        for k in filter(kk->kk∉iᵣ,_N)
            for _iᵣ in iᵣ
                jiᵣ = filter(((j,i),)->i==_iᵣ&&i!=j,_L)
                iᵣj = filter(((i,j),)->i==_iᵣ&&i!=j,_L)
                if !(isempty(jiᵣ) && isempty(iᵣj))
                    JuMP.@constraint(
                        model,
                        sum(f_rad[(k,j,i)] for (j,i) in jiᵣ) -
                        sum(f_rad[(k,i,j)] for (i,j) in iᵣj)
                        ==
                        -1.0
                    )
                end
            end

            jk = filter(((j,i),)->i==k&&i!=j,_L′)
            kj = filter(((i,j),)->i==k&&i!=j,_L′)
            if !(isempty(jk) && isempty(kj))
                JuMP.@constraint(
                    model,
                    sum(f_rad[(k,j,k)] for (j,i) in jk) -
                    sum(f_rad[(k,k,j)] for (i,j) in kj)
                    ==
                    1.0
                )
            end

            for i in filter(kk->kk∉iᵣ&&kk!=k,_N)
                ji = filter(((j,ii),)->ii==i&&ii!=j,_L′)
                ij = filter(((ii,j),)->ii==i&&ii!=j,_L′)
                if !(isempty(ji) && isempty(ij))
                    JuMP.@constraint(
                        model,
                        sum(f_rad[(k,j,i)] for (j,ii) in ji) -
                        sum(f_rad[(k,i,j)] for (ii,j) in ij)
                        ==
                        0.0
                    )
                end
            end

            for (i,j) in _L
                JuMP.@constraint(model, f_rad[(k,i,j)] >= 0)
                JuMP.@constraint(model, f_rad[(k,i,j)] <= λ[(i,j)])
                JuMP.@constraint(model, f_rad[(k,j,i)] >= 0)
                JuMP.@constraint(model, f_rad[(k,j,i)] <= λ[(j,i)])
            end
        end
    end

    # constraint_isolate_block
    for (s, switch) in ref[:switch_dispatchable]
        z_block_fr = z_block[ref[:bus_block_map][switch["f_bus"]]]
        z_block_to = z_block[ref[:bus_block_map][switch["t_bus"]]]

        γ = z_switch[s]
        JuMP.@constraint(model,  (z_block_fr - z_block_to) <=  (1-γ))
        JuMP.@constraint(model,  (z_block_fr - z_block_to) >= -(1-γ))
    end

    for b in keys(ref[:blocks])
        n_gen = length(ref[:block_gens][b])
        n_strg = length(ref[:block_storages][b])
        n_neg_loads = length([_b for (_b,ls) in ref[:block_loads] if any(any(ref[:load][l]["pd"] .< 0) for l in ls)])
        JuMP.@constraint(model, z_block[b] <= n_gen + n_strg + n_neg_loads + sum(z_switch[s] for s in keys(ref[:block_switches]) if s in keys(ref[:switch_dispatchable])))
    end

    for (i,switch) in ref[:switch]
        f_bus_id = switch["f_bus"]
        t_bus_id = switch["t_bus"]
        f_connections = switch["f_connections"]
        t_connections = switch["t_connections"]
        f_idx = (i, f_bus_id, t_bus_id)

        f_bus = ref[:bus][f_bus_id]
        t_bus = ref[:bus][t_bus_id]

        f_vmax = f_bus["vmax"][[findfirst(isequal(c), f_bus["terminals"]) for c in f_connections]]
        t_vmax = t_bus["vmax"][[findfirst(isequal(c), t_bus["terminals"]) for c in t_connections]]

        vmax = min.(fill(2.0, length(f_bus["vmax"])), f_vmax, t_vmax)

        rating = min.(fill(1.0, length(f_connections)), PMD._calc_branch_power_max_frto(switch, f_bus, t_bus)...)

        w_fr = w[f_bus_id]
        w_to = w[f_bus_id]

        # constraint_mc_switch_state_open_close
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
            JuMP.@constraint(model, w_fr[fc] - w_to[tc] <=  vmax[idx].^2 * (1-z_switch[i]))
            JuMP.@constraint(model, w_fr[fc] - w_to[tc] >= -vmax[idx].^2 * (1-z_switch[i]))
        end

        for (idx, c) in enumerate(f_connections)
            JuMP.@constraint(model, psw[f_idx][c] <=  rating[idx] * z_switch[i])
            JuMP.@constraint(model, psw[f_idx][c] >= -rating[idx] * z_switch[i])
            JuMP.@constraint(model, qsw[f_idx][c] <=  rating[idx] * z_switch[i])
            JuMP.@constraint(model, qsw[f_idx][c] >= -rating[idx] * z_switch[i])
        end

        # constraint_mc_switch_ampacity
        if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf)
            c_rating = switch["current_rating"]

            psw_fr = [psw[f_idx][c] for c in f_connections]
            qsw_fr = [qsw[f_idx][c] for c in f_connections]
            w_fr = [w[f_idx[2]][c] for c in f_connections]

            psw_sqr_fr = var_scen["psw_sqr_fr"][i]
            qsw_sqr_fr = var_scen["qsw_sqr_fr"][i]

            for (idx,c) in enumerate(f_connections)
                if isfinite(c_rating[idx])
                    p_lb, p_ub = IM.variable_domain(psw_fr[idx])
                    q_lb, q_ub = IM.variable_domain(qsw_fr[idx])
                    w_ub = IM.variable_domain(w_fr[idx])[2]

                    if (!isfinite(p_lb) || !isfinite(p_ub)) && isfinite(w_ub)
                        p_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        p_lb = -p_ub
                    end
                    if (!isfinite(q_lb) || !isfinite(q_ub)) && isfinite(w_ub)
                        q_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
                        q_lb = -q_ub
                    end

                    all(isfinite(b) for b in [p_lb, p_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, psw_fr[idx], psw_sqr_fr[idx], [p_lb, p_ub], false)
                    all(isfinite(b) for b in [q_lb, q_ub]) && PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, qsw_fr[idx], qsw_sqr_fr[idx], [q_lb, q_ub], false)
                end
            end
        end
    end

    # transformer constraints
    for (trans_id,transformer) in ref[:transformer]
        f_bus = transformer["f_bus"]
        t_bus = transformer["t_bus"]
        f_idx = (trans_id, f_bus, t_bus)
        t_idx = (trans_id, t_bus, f_bus)
        configuration = transformer["configuration"]
        f_connections = transformer["f_connections"]
        t_connections = transformer["t_connections"]
        tm_set = transformer["tm_set"]
        tm_fixed = transformer["tm_fix"]
        tm_scale = PMD.calculate_tm_scale(transformer, ref[:bus][f_bus], ref[:bus][t_bus])
        pol = transformer["polarity"]

        if configuration == PMD.WYE
            tm = var_scen["tm"][trans_id]

            p_fr = [pt[f_idx][p] for p in f_connections]
            p_to = [pt[t_idx][p] for p in t_connections]
            q_fr = [qt[f_idx][p] for p in f_connections]
            q_to = [qt[t_idx][p] for p in t_connections]

            w_fr = w[f_bus]
            w_to = w[t_bus]

            tmsqr = var_scen["tmsqr"][trans_id]

            for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
                if tm_fixed[idx]
                    JuMP.@constraint(model, w_fr[fc] == (pol*tm_scale*tm[idx])^2*w_to[tc])
                else
                    PMD.PolyhedralRelaxations.construct_univariate_relaxation!(
                        model,
                        x->x^2,
                        tm[idx],
                        tmsqr[idx],
                        [
                            JuMP.has_lower_bound(tm[idx]) ? JuMP.lower_bound(tm[idx]) : 0.9,
                            JuMP.has_upper_bound(tm[idx]) ? JuMP.upper_bound(tm[idx]) : 1.1
                        ],
                        false
                    )

                    tmsqr_w_to = var_scen["tmsqr_w_to"][trans_id]
                    PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(
                        model,
                        tmsqr[idx],
                        w_to[tc],
                        tmsqr_w_to,
                        [JuMP.lower_bound(tmsqr[idx]), JuMP.upper_bound(tmsqr[idx])],
                        [
                            JuMP.has_lower_bound(w_to[tc]) ? JuMP.lower_bound(w_to[tc]) : 0.0,
                            JuMP.has_upper_bound(w_to[tc]) ? JuMP.upper_bound(w_to[tc]) : 1.1^2
                        ]
                    )

                    JuMP.@constraint(model, w_fr[fc] == (pol*tm_scale)^2*tmsqr_w_to)
                end
            end

            JuMP.@constraint(model, p_fr + p_to .== 0)
            JuMP.@constraint(model, q_fr + q_to .== 0)

        elseif configuration == PMD.DELTA
            tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[fc] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]
            nph = length(tm_set)

            p_fr = [pt[f_idx][p] for p in f_connections]
            p_to = [pt[t_idx][p] for p in t_connections]
            q_fr = [qt[f_idx][p] for p in f_connections]
            q_to = [qt[t_idx][p] for p in t_connections]

            w_fr = w[f_bus]
            w_to = w[t_bus]

            for (idx,(fc, tc)) in enumerate(zip(f_connections,t_connections))
                # rotate by 1 to get 'previous' phase
                # e.g., for nph=3: 1->3, 2->1, 3->2
                jdx = (idx-1+1)%nph+1
                fd = f_connections[jdx]
                JuMP.@constraint(model, 3.0*(w_fr[fc] + w_fr[fd]) == 2.0*(pol*tm_scale*tm[idx])^2*w_to[tc])
            end

            for (idx,(fc, tc)) in enumerate(zip(f_connections,t_connections))
                # rotate by nph-1 to get 'previous' phase
                # e.g., for nph=3: 1->3, 2->1, 3->2
                jdx = (idx-1+nph-1)%nph+1
                fd = f_connections[jdx]
                td = t_connections[jdx]
                JuMP.@constraint(model, 2*p_fr[fc] == -(p_to[tc]+p_to[td])+(q_to[td]-q_to[tc])/sqrt(3.0))
                JuMP.@constraint(model, 2*q_fr[fc] ==  (p_to[tc]-p_to[td])/sqrt(3.0)-(q_to[td]+q_to[tc]))
            end
        end
    end

    # objective
    if !feas_chck
        delta_sw_state = var_scen["delta_sw_state"]

        for (s,switch) in ref[:switch_dispatchable]
            JuMP.@constraint(model, delta_sw_state[s] >=  (switch["state"] - z_switch[s]))
            JuMP.@constraint(model, delta_sw_state[s] >= -(switch["state"] - z_switch[s]))
        end
    end
end


## perform feasibility check by fixing common variables
function feasibility_check(scenario::Vector{Int}, all_var_common_soln::Dict{Any, Any}, load_factor::Dict{Int64, Dict{Any, Any}}; n_inf::Int=1)

    infeas_idx = []
    for scen in scenario
        var_scen = Dict()
        model_scen = JuMP.Model()
        JuMP.set_optimizer(model_scen, solver)
        variable_model(model_scen,var_scen,scen,load_factor[scen]; feas_chck=true)
        constraint_model(model_scen,var_scen,all_var_common_soln,load_factor[scen]; feas_chck=true)
        JuMP.optimize!(model_scen)
        sts = string(JuMP.termination_status(model_scen))
        if sts!="OPTIMAL"
            push!(infeas_idx,scen)
        end
        println("$scen $sts")
    end
    inf_scen = length(infeas_idx)>0 ? infeas_idx[1:n_inf] : []

    return inf_scen

end


## build bus shunt admittance matrices
function build_bus_shunt_matrices(ref::Dict{Symbol, Any}, terminals::Vector{Int}, bus_shunts::Vector{Tuple{Int64, Vector{Int64}}})
    ncnds = length(terminals)
    Gs = fill(0.0, ncnds, ncnds)
    Bs = fill(0.0, ncnds, ncnds)
    for (i, connections) in bus_shunts
        shunt = ref[:shunt][i]
        for (idx,c) in enumerate(connections)
            for (jdx,d) in enumerate(connections)
                Gs[findfirst(isequal(c), terminals),findfirst(isequal(d), terminals)] += shunt["gs"][idx,jdx]
                Bs[findfirst(isequal(c), terminals),findfirst(isequal(d), terminals)] += shunt["bs"][idx,jdx]
            end
        end
    end

    return (Gs, Bs)
end
