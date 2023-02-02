## check constraint violations by running power flow
function solve_pf(z_inverter,z_switch,load_factor)

    # create empty model
    model = JuMP.Model()
    solver = JuMP.optimizer_with_attributes(
        HiGHS.Optimizer,
        "presolve"=>"off",
        "primal_feasibility_tolerance"=>1e-6,
        "dual_feasibility_tolerance"=>1e-6,
        "mip_feasibility_tolerance"=>1e-4,
        "mip_rel_gap"=>1e-4,
        "small_matrix_value"=>1e-8,
        "allow_unbounded_or_infeasible"=>true,
        "log_to_console"=>false,
        "output_flag"=>false
    )
    JuMP.set_optimizer(model, solver)

    # variable_block_indicator
    z_block = JuMP.@variable(
        model,
        [i in keys(ref[:blocks])],
        base_name="0_z_block",
        lower_bound=0,
        upper_bound=1,
        binary=true
    )

    # variable_mc_bus_voltage_on_off -> variable_mc_bus_voltage_magnitude_sqr_on_off
    w = Dict(
        i => JuMP.@variable(
            model,
            [t in bus["terminals"]],
            base_name="0_w_$(i)",
            lower_bound=0,
        ) for (i,bus) in ref[:bus]
    )

    # w bounds
    for (i,bus) in ref[:bus]
        for (idx,t) in enumerate(bus["terminals"])
            JuMP.set_upper_bound(w[i][t], 10^2)
        end
    end

    # variable_mc_branch_power
    p = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in branch_connections[(l,i,j)]],
                base_name="0_p_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_branch]
        )
    )
    q = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in branch_connections[(l,i,j)]],
                base_name="0_q_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_branch]
        )
    )

    # p and q bounds
    for (l,i,j) in ref[:arcs_branch]
        smax = 1e5
        for (idx, c) in enumerate(branch_connections[(l,i,j)])
            PMD.set_upper_bound(p[(l,i,j)][c],  smax)
            PMD.set_lower_bound(p[(l,i,j)][c], -smax)

            PMD.set_upper_bound(q[(l,i,j)][c],  smax)
            PMD.set_lower_bound(q[(l,i,j)][c], -smax)
        end
    end

    # variable_mc_switch_power
    psw = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in switch_arc_connections[(l,i,j)]],
                base_name="0_psw_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_switch]
        )
    )

    qsw = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in switch_arc_connections[(l,i,j)]],
                base_name="0_qsw_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_switch]
        )
    )

    # psw and qsw bounds
    for (l,i,j) in ref[:arcs_switch]
        smax = 1e5
        for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
            PMD.set_upper_bound(psw[(l,i,j)][c],  smax)
            PMD.set_lower_bound(psw[(l,i,j)][c], -smax)

            PMD.set_upper_bound(qsw[(l,i,j)][c],  smax)
            PMD.set_lower_bound(qsw[(l,i,j)][c], -smax)
        end
    end

    # this explicit type erasure is necessary
    psw_expr_from = Dict( (l,i,j) => psw[(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] )
    psw_expr = merge(psw_expr_from, Dict( (l,j,i) => -1.0.*psw[(l,i,j)] for (l,i,j) in ref[:arcs_switch_from]))
    psw_auxes = Dict(
        (l,i,j) => JuMP.@variable(
            model, [c in switch_arc_connections[(l,i,j)]],
            base_name="0_psw_aux_$((l,i,j))"
        ) for (l,i,j) in ref[:arcs_switch]
    )

    qsw_expr_from = Dict( (l,i,j) => qsw[(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] )
    qsw_expr = merge(qsw_expr_from, Dict( (l,j,i) => -1.0.*qsw[(l,i,j)] for (l,i,j) in ref[:arcs_switch_from]))
    qsw_auxes = Dict(
        (l,i,j) => JuMP.@variable(
            model, [c in switch_arc_connections[(l,i,j)]],
            base_name="0_qsw_aux_$((l,i,j))"
        ) for (l,i,j) in ref[:arcs_switch]
    )

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

    # variable_mc_transformer_power
    pt = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in transformer_connections[(l,i,j)]],
                base_name="0_pt_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_transformer]
        )
    )

    qt = Dict(
        Dict(
            (l,i,j) => JuMP.@variable(
                model,
                [c in transformer_connections[(l,i,j)]],
                base_name="0_qt_($l,$i,$j)"
            ) for (l,i,j) in ref[:arcs_transformer]
        )
    )

    # variable_mc_oltc_transformer_tap
    tap = Dict(
        i => JuMP.@variable(
            model,
            [p in 1:length(ref[:transformer][i]["f_connections"])],
            base_name="0_tm_$(i)",
        ) for i in keys(filter(x->!all(x.second["tm_fix"]), ref[:transformer]))
    )

    # tap bounds
    for tr_id in p_oltc_ids, p in 1:length(ref[:transformer][tr_id]["f_connections"])
        PMD.set_lower_bound(tap[tr_id][p], ref[:transformer][tr_id]["tm_lb"][p])
        PMD.set_upper_bound(tap[tr_id][p], ref[:transformer][tr_id]["tm_ub"][p])
    end

    # variable_mc_generator_power_on_off
    pg = Dict(
        i => JuMP.@variable(
            model,
            [c in gen["connections"]],
            base_name="0_pg_$(i)",
        ) for (i,gen) in ref[:gen]
    )

    qg = Dict(
        i => JuMP.@variable(
            model,
            [c in gen["connections"]],
            base_name="0_qg_$(i)",
        ) for (i,gen) in ref[:gen]
    )

    # variable_mc_storage_power_on_off and variable_mc_storage_power_control_imaginary_on_off
    ps = Dict(
        i => JuMP.@variable(
            model,
            [c in ref[:storage][i]["connections"]],
            base_name="0_ps_$(i)",
        ) for i in keys(ref[:storage])
    )

    qs = Dict(
        i => JuMP.@variable(
            model,
            [c in ref[:storage][i]["connections"]],
            base_name="0_qs_$(i)",
        ) for i in keys(ref[:storage])
    )

    qsc = JuMP.@variable(
        model,
        [i in keys(ref[:storage])],
        base_name="0_qsc_$(i)"
    )

    # qsc bounds
    for (i,strg) in ref[:storage]
       if isfinite(sum(storage_inj_lb[i])) || haskey(strg, "qmin")
           lb = max(sum(storage_inj_lb[i]), sum(get(strg, "qmin", -Inf)))
            JuMP.set_lower_bound(qsc[i], min(lb, 0.0))
       end
       if isfinite(sum(storage_inj_ub[i])) || haskey(strg, "qmax")
           ub = min(sum(storage_inj_ub[i]), sum(get(strg, "qmax", Inf)))
            JuMP.set_upper_bound(qsc[i], max(ub, 0.0))
       end
   end

    # variable_storage_energy, variable_storage_charge and variable_storage_discharge
    se = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_se",
        lower_bound = 0.0,
    )

    sc = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sc",
        lower_bound = 0.0,
    )

    sd = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sd",
        lower_bound = 0.0,
    )

    # variable_storage_complementary_indicator and variable_storage_complementary_indicator
    sc_on = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sc_on",
        binary = true,
        lower_bound=0,
        upper_bound=1
    )

    sd_on = JuMP.@variable(model,
        [i in keys(ref[:storage])],
        base_name="0_sd_on",
        binary = true,
        lower_bound=0,
        upper_bound=1
    )

    # load variables
    pd = Dict()
    qd = Dict()
    pd_bus = Dict()
    qd_bus = Dict()
    (Xdr,Xdi) = PMD.variable_mx_complex(model, load_del_ids, load_connections, load_connections; symm_bound=bound, name="0_Xd")
    (CCdr, CCdi) = PMD.variable_mx_hermitian(model, load_del_ids, load_connections; sqrt_upper_bound=cmax, sqrt_lower_bound=cmin, name="0_CCd")

    for i in intersect(load_wye_ids, load_cone_ids)
        load = ref[:load][i]
        bus = ref[:bus][load["load_bus"]]
        pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
        pd[i] = JuMP.@variable(
            model,
            [c in load_connections[i]],
            base_name="0_pd_$(i)"
        )
        qd[i] = JuMP.@variable(
            model,
            [c in load_connections[i]],
            base_name="0_qd_$(i)"
        )

        load = ref[:load][i]
        bus = ref[:bus][load["load_bus"]]
        pmin = -1e5
        pmax = 1e5
        qmin = -1e5
        qmax = 1e5
        for (idx,c) in enumerate(load_connections[i])
            PMD.set_lower_bound(pd[i][c], pmin)
            PMD.set_upper_bound(pd[i][c], pmax)
            PMD.set_lower_bound(qd[i][c], qmin)
            PMD.set_upper_bound(qd[i][c], qmax)
        end
    end

    # variable_mc_capacitor_switch_state
    z_cap = Dict(
        i => JuMP.@variable(
            model,
            [p in cap["connections"]],
            base_name="0_cap_sw_$(i)",
            binary = true,
        ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
    )

    # variable_mc_capacitor_reactive_power
    qc = Dict(
        i => JuMP.@variable(
            model,
            [p in cap["connections"]],
            base_name="0_cap_cur_$(i)",
        ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
    )

    # voltage sources are always grid-forming
    for ((t,j), z_inv) in z_inverter
        if t == :gen && startswith(ref[t][j]["source_id"], "voltage_source")
            JuMP.@constraint(model, z_inv == z_block[ref[:bus_block_map][ref[t][j]["$(t)_bus"]]])
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
        load_scen = deepcopy(load)
        load_scen["pd"] = load["pd"]*load_factor[load_id]
        load_scen["qd"] = load["qd"]*load_factor[load_id]
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
            Td = [1 -1 0; 0 1 -1; -1 0 1]
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

        pd_zblock = Dict(l => JuMP.@variable(model, [c in conns], base_name="0_pd_zblock_$(l)") for (l,conns) in ref[:bus_conns_load][i])
        qd_zblock = Dict(l => JuMP.@variable(model, [c in conns], base_name="0_qd_zblock_$(l)") for (l,conns) in ref[:bus_conns_load][i])

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
            qsc_zblock = JuMP.@variable(model, base_name="0_qd_zblock_$(i)")

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

    # switch constraints
    for (i,switch) in ref[:switch]
        f_bus_id = switch["f_bus"]
        t_bus_id = switch["t_bus"]
        f_connections = switch["f_connections"]
        t_connections = switch["t_connections"]
        f_idx = (i, f_bus_id, t_bus_id)

        w_fr = w[f_bus_id]
        w_to = w[f_bus_id]

        f_bus = ref[:bus][f_bus_id]
        t_bus = ref[:bus][t_bus_id]

        f_vmax = f_bus["vmax"][[findfirst(isequal(c), f_bus["terminals"]) for c in f_connections]]
        t_vmax = t_bus["vmax"][[findfirst(isequal(c), t_bus["terminals"]) for c in t_connections]]

        vmax = min.(fill(1e5, length(f_bus["vmax"])), f_vmax, t_vmax)

        # constraint_mc_switch_state_open_close
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
            JuMP.@constraint(model, w_fr[fc] - w_to[tc] <=  vmax[idx].^2 * (1-z_switch[i]))
            JuMP.@constraint(model, w_fr[fc] - w_to[tc] >= -vmax[idx].^2 * (1-z_switch[i]))
        end

        rating = min.(fill(1e5, length(f_connections)), PMD._calc_branch_power_max_frto(switch, f_bus, t_bus)...)

        for (idx, c) in enumerate(f_connections)
            JuMP.@constraint(model, psw[f_idx][c] <=  rating[idx] * z_switch[i])
            JuMP.@constraint(model, psw[f_idx][c] >= -rating[idx] * z_switch[i])
            JuMP.@constraint(model, qsw[f_idx][c] <=  rating[idx] * z_switch[i])
            JuMP.@constraint(model, qsw[f_idx][c] >= -rating[idx] * z_switch[i])
        end
    end

    # transformer constraints
    for (i,transformer) in ref[:transformer]
        f_bus = transformer["f_bus"]
        t_bus = transformer["t_bus"]
        f_idx = (i, f_bus, t_bus)
        t_idx = (i, t_bus, f_bus)
        configuration = transformer["configuration"]
        f_connections = transformer["f_connections"]
        t_connections = transformer["t_connections"]
        tm_set = transformer["tm_set"]
        tm_fixed = transformer["tm_fix"]
        tm_scale = PMD.calculate_tm_scale(transformer, ref[:bus][f_bus], ref[:bus][t_bus])
        pol = transformer["polarity"]

        if configuration == PMD.WYE
            tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[idx] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]

            p_fr = [pt[f_idx][p] for p in f_connections]
            p_to = [pt[t_idx][p] for p in t_connections]
            q_fr = [qt[f_idx][p] for p in f_connections]
            q_to = [qt[t_idx][p] for p in t_connections]

            w_fr = w[f_bus]
            w_to = w[t_bus]

            tmsqr = [
                tm_fixed[i] ? tm[i]^2 : JuMP.@variable(
                    model,
                    base_name="0_tmsqr_$(trans_id)_$(f_connections[i])",
                    start=JuMP.start_value(tm[i])^2,
                    lower_bound=JuMP.has_lower_bound(tm[i]) ? JuMP.lower_bound(tm[i])^2 : 0.9^2,
                    upper_bound=JuMP.has_upper_bound(tm[i]) ? JuMP.upper_bound(tm[i])^2 : 1.1^2
                ) for i in 1:length(tm)
            ]

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

                    tmsqr_w_to = JuMP.@variable(model, base_name="0_tmsqr_w_to_$(trans_id)_$(t_bus)_$(tc)")
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

    ## solve manual model
    JuMP.optimize!(model)
    sts = string(JuMP.termination_status(model))
    # println("Substation pg: $([JuMP.value(pg[4][c]) for c=1:3])")

    ## calculate number of constraint violations
    w_viol = sum(sum(JuMP.value(w[i][t])>bus["vmax"][idx]^2 for (idx,t) in enumerate(bus["terminals"])) for (i,bus) in ref[:bus])
    pg_viol = sum(sum(JuMP.value(pg[i][c])>gen["pmax"][idx] for (idx,c) in enumerate(gen["connections"]))
                + sum(JuMP.value(pg[i][c])<min(0.0, gen["pmin"][idx]) for (idx,c) in enumerate(gen["connections"]))
            for (i,gen) in ref[:gen])
    qg_viol = sum(sum(JuMP.value(qg[i][c])>gen["qmax"][idx] for (idx,c) in enumerate(gen["connections"]))
                + sum(JuMP.value(qg[i][c])<min(0.0, gen["qmin"][idx]) for (idx,c) in enumerate(gen["connections"]))
            for (i,gen) in ref[:gen])
    p_q_viol = 0
    # for (i,branch) in ref[:branch]
    #     f_connections = branch["f_connections"]
    #     t_connections = branch["t_connections"]
    #     f_idx = (i, branch["f_bus"], branch["t_bus"])
    #     t_idx = (i, branch["t_bus"], branch["f_bus"])
    #     if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf)
    #         c_rating_sqr = branch["c_rating_a"].^2
    #         p_fr = [JuMP.value(p[f_idx][c]) for c in f_connections]
    #         q_fr = [JuMP.value(q[f_idx][c]) for c in f_connections]
    #         w_fr = [JuMP.value(w[f_idx[2]][c]) for c in f_connections]

    #         p_to = [JuMP.value(p[t_idx][c]) for c in t_connections]
    #         q_to = [JuMP.value(q[t_idx][c]) for c in t_connections]
    #         w_to = [JuMP.value(w[t_idx[2]][c]) for c in t_connections]

    #         p_q_viol += sum(p_fr.^2 + q_fr.^2 .> w_fr.*c_rating_sqr) + sum(p_to.^2 + q_to.^2 .> w_to.*c_rating_sqr)
    #     end
    # end

    pt_viol = 0
    qt_viol = 0
    for arc in ref[:arcs_transformer_from]
        (l,i,j) = arc
        rate_a_fr, rate_a_to = PMD._calc_transformer_power_ub_frto(ref[:transformer][l], ref[:bus][i], ref[:bus][j])
        pt_viol += sum(JuMP.value(pt[(l,i,j)][fc])<-rate_a_fr[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(pt[(l,i,j)][fc])> rate_a_fr[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(pt[(l,i,j)][tc])<-rate_a_to[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(pt[(l,i,j)][tc])> rate_a_to[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
        qt_viol += sum(JuMP.value(qt[(l,i,j)][fc])<-rate_a_fr[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(qt[(l,i,j)][fc])> rate_a_fr[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(qt[(l,i,j)][tc])<-rate_a_to[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
                 + sum(JuMP.value(qt[(l,i,j)][tc])> rate_a_to[idx] for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)])))
    end

    N = w_viol + p_q_viol + pg_viol + qg_viol + pt_viol + qt_viol
    return N

end

## setup all scenario model and solve opf
function solve_model1()

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

    # Generate scenarios
    N_scenarios=2
    obj_fac = [1.21; 1.25]
    # load_factor = load_uncertainty(N_scenarios, 0.10)
    load_factor = JLD.load("load_factor.jld")["load_factor"]

    # setup and solve model adding one scenario in each iteration
    scenarios = [1;2]
    viol_ind = true
    while length(scenarios)<=N_scenarios && viol_ind

        # create empty model
        model = JuMP.Model()
        JuMP.set_optimizer(model, solver)

        # variable_block_indicator
        z_block = Dict(scen => JuMP.@variable(
            model,
            [i in keys(ref[:blocks])],
            base_name="0_z_block_$(scen)",
            lower_bound=0,
            upper_bound=1,
            binary=true
        ) for scen in scenarios)

        # variable_inverter_indicator
        global z_inverter = Dict(scen => Dict(
            (t,i) => get(ref[t][i], "inverter", 1) == 1 ? JuMP.@variable(
                model,
                base_name="0_$(t)_z_inverter_$(i)_$(scen)",
                binary=true,
                lower_bound=0,
                upper_bound=1,
            ) : 0 for t in [:storage, :gen] for i in keys(ref[t])
        ) for scen in scenarios)

        # variable_mc_bus_voltage_on_off -> variable_mc_bus_voltage_magnitude_sqr_on_off
        w = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [t in bus["terminals"]],
                base_name="0_w_$(i)_$(scen)",
                lower_bound=0,
            ) for (i,bus) in ref[:bus]
        ) for scen in scenarios)

        # w bounds
        for (i,bus) in ref[:bus]
            for (idx,t) in enumerate(bus["terminals"])
                for scen in scenarios
                    isfinite(bus["vmax"][idx]) && JuMP.set_upper_bound(w[scen][i][t], bus["vmax"][idx]^2)
                end
            end
        end

        # variable_mc_branch_power
        p = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in branch_connections[(l,i,j)]],
                    base_name="0_p_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_branch]
            )
        for scen in scenarios)
        q = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in branch_connections[(l,i,j)]],
                    base_name="0_q_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_branch]
            )
        for scen in scenarios)

        # p and q bounds
        for (l,i,j) in ref[:arcs_branch]
            smax = PMD._calc_branch_power_max(ref[:branch][l], ref[:bus][i])
            for (idx, c) in enumerate(branch_connections[(l,i,j)])
                for scen in scenarios
                    PMD.set_upper_bound(p[scen][(l,i,j)][c],  smax[idx])
                    PMD.set_lower_bound(p[scen][(l,i,j)][c], -smax[idx])

                    PMD.set_upper_bound(q[scen][(l,i,j)][c],  smax[idx])
                    PMD.set_lower_bound(q[scen][(l,i,j)][c], -smax[idx])
                end
            end
        end

        # variable_mc_switch_power
        psw = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in switch_arc_connections[(l,i,j)]],
                    base_name="0_psw_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_switch]
            )
        for scen in scenarios)

        qsw = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in switch_arc_connections[(l,i,j)]],
                    base_name="0_qsw_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_switch]
            )
        for scen in scenarios)

        # psw and qsw bounds
        for (l,i,j) in ref[:arcs_switch]
            smax = PMD._calc_branch_power_max(ref[:switch][l], ref[:bus][i])
            for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
                for scen in scenarios
                    PMD.set_upper_bound(psw[scen][(l,i,j)][c],  smax[idx])
                    PMD.set_lower_bound(psw[scen][(l,i,j)][c], -smax[idx])

                    PMD.set_upper_bound(qsw[scen][(l,i,j)][c],  smax[idx])
                    PMD.set_lower_bound(qsw[scen][(l,i,j)][c], -smax[idx])
                end
            end
        end

        # this explicit type erasure is necessary
        psw_expr_from = Dict(scen => Dict( (l,i,j) => psw[scen][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] ) for scen in scenarios)
        psw_expr = Dict(scen => merge(psw_expr_from[scen], Dict( (l,j,i) => -1.0.*psw[scen][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from])) for scen in scenarios)
        psw_auxes = Dict(scen => Dict(
            (l,i,j) => JuMP.@variable(
                model, [c in switch_arc_connections[(l,i,j)]],
                base_name="0_psw_aux_$((l,i,j))_$(scen)"
            ) for (l,i,j) in ref[:arcs_switch]
        ) for scen in scenarios)

        qsw_expr_from = Dict(scen => Dict( (l,i,j) => qsw[scen][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from] ) for scen in scenarios)
        qsw_expr = Dict(scen => merge(qsw_expr_from[scen], Dict( (l,j,i) => -1.0.*qsw[scen][(l,i,j)] for (l,i,j) in ref[:arcs_switch_from])) for scen in scenarios)
        qsw_auxes = Dict(scen => Dict(
            (l,i,j) => JuMP.@variable(
                model, [c in switch_arc_connections[(l,i,j)]],
                base_name="0_qsw_aux_$((l,i,j))_$(scen)"
            ) for (l,i,j) in ref[:arcs_switch]
        ) for scen in scenarios)

        # This is needed to get around error: "unexpected affine expression in nlconstraint" and overwrite psw/qsw
        for scen in scenarios
            for ((l,i,j), psw_aux) in psw_auxes[scen]
                for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
                    JuMP.@constraint(model, psw_expr[scen][(l,i,j)][c] == psw_aux[c])
                end
            end
            for (k,psw_aux) in psw_auxes[scen]
                psw[scen][k] = psw_aux
            end

            for ((l,i,j), qsw_aux) in qsw_auxes[scen]
                for (idx, c) in enumerate(switch_arc_connections[(l,i,j)])
                    JuMP.@constraint(model, qsw_expr[scen][(l,i,j)][c] == qsw_aux[c])
                end
            end
            for (k,qsw_aux) in qsw_auxes[scen]
                qsw[scen][k] = qsw_aux
            end
        end

        # variable_switch_state
        z_switch = Dict(scen => Dict(i => JuMP.@variable(
            model,
            base_name="0_switch_state_$(i)_$(scen)",
            binary=true,
            lower_bound=0,
            upper_bound=1,
        ) for i in keys(ref[:switch_dispatchable])) for scen in scenarios)

        # fixed switches
        for i in [i for i in keys(ref[:switch]) if !(i in keys(ref[:switch_dispatchable]))]
            for scen in scenarios
                z_switch[scen][i] = ref[:switch][i]["state"]
            end
        end

        # variable_mc_transformer_power
        pt = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in transformer_connections[(l,i,j)]],
                    base_name="0_pt_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_transformer]
            )
        for scen in scenarios)

        qt = Dict(scen =>
            Dict(
                (l,i,j) => JuMP.@variable(
                    model,
                    [c in transformer_connections[(l,i,j)]],
                    base_name="0_qt_($l,$i,$j)_$(scen)"
                ) for (l,i,j) in ref[:arcs_transformer]
            )
        for scen in scenarios)

        # pt and qt bounds
        for arc in ref[:arcs_transformer_from]
            (l,i,j) = arc
            rate_a_fr, rate_a_to = PMD._calc_transformer_power_ub_frto(ref[:transformer][l], ref[:bus][i], ref[:bus][j])

            for (idx, (fc,tc)) in enumerate(zip(transformer_connections[(l,i,j)], transformer_connections[(l,j,i)]))
                for scen in scenarios
                    PMD.set_lower_bound(pt[scen][(l,i,j)][fc], -rate_a_fr[idx])
                    PMD.set_upper_bound(pt[scen][(l,i,j)][fc],  rate_a_fr[idx])
                    PMD.set_lower_bound(pt[scen][(l,j,i)][tc], -rate_a_to[idx])
                    PMD.set_upper_bound(pt[scen][(l,j,i)][tc],  rate_a_to[idx])

                    PMD.set_lower_bound(qt[scen][(l,i,j)][fc], -rate_a_fr[idx])
                    PMD.set_upper_bound(qt[scen][(l,i,j)][fc],  rate_a_fr[idx])
                    PMD.set_lower_bound(qt[scen][(l,j,i)][tc], -rate_a_to[idx])
                    PMD.set_upper_bound(qt[scen][(l,j,i)][tc],  rate_a_to[idx])
                end
            end
        end

        # variable_mc_oltc_transformer_tap
        tap = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [p in 1:length(ref[:transformer][i]["f_connections"])],
                base_name="0_tm_$(i)_$(scen)",
            ) for i in keys(filter(x->!all(x.second["tm_fix"]), ref[:transformer]))
        ) for scen in scenarios)

        # tap bounds
        for tr_id in p_oltc_ids, p in 1:length(ref[:transformer][tr_id]["f_connections"])
            for scen in scenarios
                PMD.set_lower_bound(tap[scen][tr_id][p], ref[:transformer][tr_id]["tm_lb"][p])
                PMD.set_upper_bound(tap[scen][tr_id][p], ref[:transformer][tr_id]["tm_ub"][p])
            end
        end

        # variable_mc_generator_power_on_off
        pg = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [c in gen["connections"]],
                base_name="0_pg_$(i)_$(scen)",
            ) for (i,gen) in ref[:gen]
        ) for scen in scenarios)

        qg = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [c in gen["connections"]],
                base_name="0_qg_$(i)_$(scen)",
            ) for (i,gen) in ref[:gen]
        ) for scen in scenarios)

        # pg and qg bounds
        for (i,gen) in ref[:gen]
            for (idx,c) in enumerate(gen["connections"])
                for scen in scenarios
                    isfinite(gen["pmin"][idx]) && JuMP.set_lower_bound(pg[scen][i][c], min(0.0, gen["pmin"][idx]))
                    isfinite(gen["pmax"][idx]) && JuMP.set_upper_bound(pg[scen][i][c], gen["pmax"][idx])

                    isfinite(gen["qmin"][idx]) && JuMP.set_lower_bound(qg[scen][i][c], min(0.0, gen["qmin"][idx]))
                    isfinite(gen["qmax"][idx]) && JuMP.set_upper_bound(qg[scen][i][c], gen["qmax"][idx])
                end
            end
        end

        # variable_mc_storage_power_on_off and variable_mc_storage_power_control_imaginary_on_off
        ps = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [c in ref[:storage][i]["connections"]],
                base_name="0_ps_$(i)_$(scen)",
            ) for i in keys(ref[:storage])
        ) for scen in scenarios)

        qs = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [c in ref[:storage][i]["connections"]],
                base_name="0_qs_$(i)_$(scen)",
            ) for i in keys(ref[:storage])
        ) for scen in scenarios)

        qsc = Dict(scen => JuMP.@variable(
            model,
            [i in keys(ref[:storage])],
            base_name="0_qsc_$(i)_$(scen)"
        ) for scen in scenarios)

        # ps, qs and qsc bounds
        for (i,strg) in ref[:storage]
             for (idx, c) in enumerate(strg["connections"])
                if !isinf(storage_inj_lb[i][idx])
                    for scen in scenarios
                        PMD.set_lower_bound(ps[scen][i][c], storage_inj_lb[i][idx])
                        PMD.set_lower_bound(qs[scen][i][c], storage_inj_lb[i][idx])
                    end
                end
                if !isinf(storage_inj_ub[i][idx])
                    for scen in scenarios
                        PMD.set_upper_bound(ps[scen][i][c], storage_inj_ub[i][idx])
                        PMD.set_upper_bound(qs[scen][i][c], storage_inj_ub[i][idx])
                    end
                end
            end

            if isfinite(sum(storage_inj_lb[i])) || haskey(strg, "qmin")
                lb = max(sum(storage_inj_lb[i]), sum(get(strg, "qmin", -Inf)))
                for scen in scenarios
                    JuMP.set_lower_bound(qsc[scen][i], min(lb, 0.0))
                end
            end
            if isfinite(sum(storage_inj_ub[i])) || haskey(strg, "qmax")
                ub = min(sum(storage_inj_ub[i]), sum(get(strg, "qmax", Inf)))
                for scen in scenarios
                    JuMP.set_upper_bound(qsc[scen][i], max(ub, 0.0))
                end
            end
        end

        # variable_storage_energy, variable_storage_charge and variable_storage_discharge
        se = Dict(scen => JuMP.@variable(model,
            [i in keys(ref[:storage])],
            base_name="0_se_$(scen)",
            lower_bound = 0.0,
        ) for scen in scenarios)

        sc = Dict(scen => JuMP.@variable(model,
            [i in keys(ref[:storage])],
            base_name="0_sc_$(scen)",
            lower_bound = 0.0,
        ) for scen in scenarios)

        sd = Dict(scen => JuMP.@variable(model,
            [i in keys(ref[:storage])],
            base_name="0_sd_$(scen)",
            lower_bound = 0.0,
        ) for scen in scenarios)

        # se, sc and sd bounds
        for (i, storage) in ref[:storage]
            for scen in scenarios
                PMD.set_upper_bound(se[scen][i], storage["energy_rating"])
                PMD.set_upper_bound(sc[scen][i], storage["charge_rating"])
                PMD.set_upper_bound(sd[scen][i], storage["discharge_rating"])
            end
        end

        # variable_storage_complementary_indicator and variable_storage_complementary_indicator
        sc_on = Dict(scen => JuMP.@variable(model,
            [i in keys(ref[:storage])],
            base_name="0_sc_on_$(scen)",
            binary = true,
            lower_bound=0,
            upper_bound=1
        ) for scen in scenarios)

        sd_on = Dict(scen => JuMP.@variable(model,
            [i in keys(ref[:storage])],
            base_name="0_sd_on_$(scen)",
            binary = true,
            lower_bound=0,
            upper_bound=1
        ) for scen in scenarios)

        # load variables
        pd = Dict(scen => Dict() for scen in scenarios)
        qd = Dict(scen => Dict() for scen in scenarios)
        pd_bus = Dict(scen => Dict() for scen in scenarios)
        qd_bus = Dict(scen => Dict() for scen in scenarios)
        if !isempty(load_del_ids)
            Xdr = Dict()
            Xdi = Dict()
            CCdr = Dict()
            CCdi = Dict()
            for scen in scenarios
                (Xdr[scen],Xdi[scen]) = PMD.variable_mx_complex(model, load_del_ids, load_connections, load_connections; symm_bound=bound, name="0_Xd_$(scen)")
                (CCdr[scen], CCdi[scen]) = PMD.variable_mx_hermitian(model, load_del_ids, load_connections; sqrt_upper_bound=cmax, sqrt_lower_bound=cmin, name="0_CCd_$(scen)")
            end
        end

        for i in intersect(load_wye_ids, load_cone_ids)
            load = ref[:load][i]
            bus = ref[:bus][load["load_bus"]]
            pmin, pmax, qmin, qmax = PMD._calc_load_pq_bounds(load, bus)
            for scen in scenarios
                pd[scen][i] = JuMP.@variable(
                    model,
                    [c in load_connections[i]],
                    base_name="0_pd_$(i)_$(scen)"
                )
                qd[scen][i] = JuMP.@variable(
                    model,
                    [c in load_connections[i]],
                    base_name="0_qd_$(i)_$(scen)"
                )

                for (idx,c) in enumerate(load_connections[i])
                    PMD.set_lower_bound(pd[scen][i][c], pmin[idx])
                    PMD.set_upper_bound(pd[scen][i][c], pmax[idx])
                    PMD.set_lower_bound(qd[scen][i][c], qmin[idx])
                    PMD.set_upper_bound(qd[scen][i][c], qmax[idx])
                end
            end
        end

        # variable_mc_capacitor_switch_state
        z_cap = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [p in cap["connections"]],
                base_name="0_cap_sw_$(i)_$(scen)",
                binary = true,
            ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
        ) for scen in scenarios)

        # variable_mc_capacitor_reactive_power
        qc = Dict(scen => Dict(
            i => JuMP.@variable(
                model,
                [p in cap["connections"]],
                base_name="0_cap_cur_$(i)_$(scen)",
            ) for (i,cap) in [(id,cap) for (id,cap) in ref[:shunt] if haskey(cap,"controls")]
        ) for scen in scenarios)

        # variable representing if switch ab has 'color' k
        y = Dict(scen => Dict() for scen in scenarios)
        for k in L
            for ab in keys(ref[:switch])
                for scen in scenarios
                    y[scen][(k,ab)] = JuMP.@variable(
                        model,
                        base_name="0_y_gfm[$k,$ab]_$(scen)",
                        binary=true,
                        lower_bound=0,
                        upper_bound=1
                    )
                end
            end
        end

        # Eqs. (9)-(10)
        f = Dict(scen => Dict() for scen in scenarios)
        ϕ = Dict(scen => Dict() for scen in scenarios)
        for kk in L # color
            for ab in keys(ref[:switch])
                for scen in scenarios
                    f[scen][(kk,ab)] = JuMP.@variable(
                        model,
                        base_name="0_f_gfm[$kk,$ab]_$(scen)"
                    )
                    JuMP.@constraint(model, f[scen][kk,ab] >= -length(keys(ref[:switch]))*(z_switch[scen][ab]))
                    JuMP.@constraint(model, f[scen][kk,ab] <=  length(keys(ref[:switch]))*(z_switch[scen][ab]))
                end
            end
            touched = Set()
            ab = 1
            for k in sort(collect(L)) # fr block
                for k′ in sort(collect(filter(x->x!=k,L))) # to block
                    if (k,k′) ∉ touched
                        push!(touched, (k,k′), (k′,k))
                        for scen in scenarios
                            ϕ[scen][(kk,ab)] = JuMP.@variable(
                                model,
                                base_name="0_phi_gfm[$kk,$ab]_$(scen)",
                                lower_bound=0,
                                upper_bound=1
                            )
                        end
                        ab += 1
                    end
                end
            end
        end

        # voltage sources are always grid-forming
        for scen in scenarios
            for ((t,j), z_inv) in z_inverter[scen]
                if t == :gen && startswith(ref[t][j]["source_id"], "voltage_source")
                    JuMP.@constraint(model, z_inv == z_block[scen][ref[:bus_block_map][ref[t][j]["$(t)_bus"]]])
                end
            end
        end

        # constrain each y to have only one color
        for ab in keys(ref[:switch])
            for scen in scenarios
                JuMP.@constraint(model, sum(y[scen][(k,ab)] for k in L) <= z_switch[scen][ab])
            end
        end

        # Eqs. (3)-(7)
        for k in L
            Dₖ = ref[:block_inverters][k]
            Tₖ = ref[:block_switches][k]

            if !isempty(Dₖ)
                # Eq. (14)
                for scen in scenarios
                    JuMP.@constraint(model, sum(z_inverter[scen][i] for i in Dₖ) >= sum(1-z_switch[scen][ab] for ab in Tₖ)-length(Tₖ)+z_block[scen][k])
                    JuMP.@constraint(model, sum(z_inverter[scen][i] for i in Dₖ) <= z_block[scen][k])
                end

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
                                for scen in scenarios
                                    JuMP.@constraint(model, ps[scen][j][c] <= pmax[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, ps[scen][j][c] <= pmax[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(qmax[idx]) && qmax[idx] >= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, qs[scen][j][c] <= qmax[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, qs[scen][j][c] <= qmax[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(pmin[idx]) && pmin[idx] <= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, ps[scen][j][c] >= pmin[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, ps[scen][j][c] >= pmin[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(qmin[idx]) && qmin[idx] <= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, qs[scen][j][c] >= qmin[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, qs[scen][j][c] >= qmin[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                        end
                    elseif t == :gen
                        pmin = ref[t][j]["pmin"]
                        pmax = ref[t][j]["pmax"]
                        qmin = ref[t][j]["qmin"]
                        qmax = ref[t][j]["qmax"]

                        for (idx,c) in enumerate(ref[t][j]["connections"])
                            if isfinite(pmax[idx]) && pmax[idx] >= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, pg[scen][j][c] <= pmax[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, pg[scen][j][c] <= pmax[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(qmax[idx]) && qmax[idx] >= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, qg[scen][j][c] <= qmax[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, qg[scen][j][c] <= qmax[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(pmin[idx]) && pmin[idx] <= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, pg[scen][j][c] >= pmin[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, pg[scen][j][c] >= pmin[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                            if isfinite(qmin[idx]) && qmin[idx] <= 0
                                for scen in scenarios
                                    JuMP.@constraint(model, qg[scen][j][c] >= qmin[idx] * (sum(z_switch[scen][ab] for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                    JuMP.@constraint(model, qg[scen][j][c] >= qmin[idx] * (sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ) + sum(z_inverter[scen][i] for i in Dₖ)))
                                end
                            end
                        end
                    end
                end
            end

            for ab in Tₖ
                for scen in scenarios
                    # Eq. (6)
                    JuMP.@constraint(model, sum(z_inverter[scen][i] for i in Dₖ) >= y[scen][(k, ab)] - (1 - z_switch[scen][ab]))
                    JuMP.@constraint(model, sum(z_inverter[scen][i] for i in Dₖ) <= y[scen][(k, ab)] + (1 - z_switch[scen][ab]))

                    # Eq. (8)
                    JuMP.@constraint(model, y[scen][(k,ab)] <= sum(z_inverter[scen][i] for i in Dₖ))
                end

                for dc in filter(x->x!=ab, Tₖ)
                    for k′ in L
                        # Eq. (7)
                        for scen in scenarios
                            JuMP.@constraint(model, y[scen][(k′,ab)] >= y[scen][(k′,dc)] - (1 - z_switch[scen][dc]) - (1 - z_switch[scen][ab]))
                            JuMP.@constraint(model, y[scen][(k′,ab)] <= y[scen][(k′,dc)] + (1 - z_switch[scen][dc]) + (1 - z_switch[scen][ab]))
                        end
                    end
                end
            end

            for scen in scenarios
                # Eq. (11)
                JuMP.@constraint(model, sum(f[scen][(k,ab)] for ab in filter(x->map_id_pairs[x][1] == k, Tₖ)) - sum(f[scen][(k,ab)] for ab in filter(x->map_id_pairs[x][2] == k, Tₖ)) + sum(ϕ[scen][(k,ab)] for ab in Φₖ[k]) == length(L) - 1)

                # Eq. (15)
                JuMP.@constraint(model, z_block[scen][k] <= sum(z_inverter[scen][i] for i in Dₖ) + sum(y[scen][(k′,ab)] for k′ in L for ab in Tₖ))
            end

            for k′ in filter(x->x!=k, L)
                Tₖ′ = ref[:block_switches][k′]
                kk′ = map_virtual_pairs_id[k][(k,k′)]

                # Eq. (12)
                for scen in scenarios
                    JuMP.@constraint(model, sum(f[scen][(k,ab)] for ab in filter(x->map_id_pairs[x][1]==k′, Tₖ′)) - sum(f[scen][(k,ab)] for ab in filter(x->map_id_pairs[x][2]==k′, Tₖ′)) - ϕ[scen][(k,(kk′))] == -1)
                end

                # Eq. (13)
                for ab in Tₖ′
                    for scen in scenarios
                        JuMP.@constraint(model, y[scen][k,ab] <= 1 - ϕ[scen][(k,kk′)])
                    end
                end
            end
        end

        # constraint_mc_inverter_theta_ref
        for (i,bus) in ref[:bus]
            # reference bus "theta" constraint
            vmax = min(bus["vmax"]..., 2.0)
            if isfinite(vmax)
                for scen in scenarios
                    if length(w[scen][i]) > 1 && !isempty([z_inverter[scen][inv_obj] for inv_obj in ref[:bus_inverters][i]])
                        for t in 2:length(w[scen][i])
                            JuMP.@constraint(model, w[scen][i][t] - w[scen][i][1] <=  vmax^2 * (1 - sum([z_inverter[scen][inv_obj] for inv_obj in ref[:bus_inverters][i]])))
                            JuMP.@constraint(model, w[scen][i][t] - w[scen][i][1] >= -vmax^2 * (1 - sum([z_inverter[scen][inv_obj] for inv_obj in ref[:bus_inverters][i]])))
                        end
                    end
                end
            end
        end

        # constraint_mc_bus_voltage_block_on_off
        for (i,bus) in ref[:bus]
            # bus voltage on off constraint
            for (idx,t) in [(idx,t) for (idx,t) in enumerate(bus["terminals"]) if !bus["grounded"][idx]]
                for scen in scenarios
                    isfinite(bus["vmax"][idx]) && JuMP.@constraint(model, w[scen][i][t] <= bus["vmax"][idx]^2*z_block[scen][ref[:bus_block_map][i]])
                    isfinite(bus["vmin"][idx]) && JuMP.@constraint(model, w[scen][i][t] >= bus["vmin"][idx]^2*z_block[scen][ref[:bus_block_map][i]])
                end
            end
        end

        # constraint_mc_generator_power_block_on_off
        for (i,gen) in ref[:gen]
            for (idx, c) in enumerate(gen["connections"])
                for scen in scenarios
                    isfinite(gen["pmin"][idx]) && JuMP.@constraint(model, pg[scen][i][c] >= gen["pmin"][idx]*z_block[scen][ref[:gen_block_map][i]])
                    isfinite(gen["qmin"][idx]) && JuMP.@constraint(model, qg[scen][i][c] >= gen["qmin"][idx]*z_block[scen][ref[:gen_block_map][i]])

                    isfinite(gen["pmax"][idx]) && JuMP.@constraint(model, pg[scen][i][c] <= gen["pmax"][idx]*z_block[scen][ref[:gen_block_map][i]])
                    isfinite(gen["qmax"][idx]) && JuMP.@constraint(model, qg[scen][i][c] <= gen["qmax"][idx]*z_block[scen][ref[:gen_block_map][i]])
                end
            end
        end

        # constraint_mc_load_power
        for (load_id,load) in ref[:load]
            bus_id = load["load_bus"]
            bus = ref[:bus][bus_id]
            Td = [1 -1 0; 0 1 -1; -1 0 1]
            load_scen = deepcopy(load)
            for scen in scenarios
                load_scen["pd"] = load["pd"]*load_factor[scen][load_id]
                load_scen["qd"] = load["qd"]*load_factor[scen][load_id]
                a, alpha, b, beta = PMD._load_expmodel_params(load_scen, bus)
                pd0 = load_scen["pd"]
                qd0 = load_scen["qd"]

                if load["configuration"]==PMD.WYE
                    if load["model"]==PMD.POWER
                        pd[scen][load_id] = JuMP.Containers.DenseAxisArray(pd0, load["connections"])
                        qd[scen][load_id] = JuMP.Containers.DenseAxisArray(qd0, load["connections"])
                    elseif load["model"]==PMD.IMPEDANCE
                        _w = w[scen][bus_id][[c for c in load["connections"]]]
                        pd[scen][load_id] = a.*_w
                        qd[scen][load_id] = b.*_w
                    else
                        for (idx,c) in enumerate(load["connections"])
                            JuMP.@constraint(model, pd[scen][load_id][c]==1/2*a[idx]*(w[scen][bus_id][c]+1))
                            JuMP.@constraint(model, qd[scen][load_id][c]==1/2*b[idx]*(w[scen][bus_id][c]+1))
                        end
                    end

                    pd_bus[scen][load_id] = pd[scen][load_id]
                    qd_bus[scen][load_id] = qd[scen][load_id]

                elseif load["configuration"]==PMD.DELTA
                    pd_bus[scen][load_id] = LinearAlgebra.diag(Xdr[scen][load_id]*Td)
                    qd_bus[scen][load_id] = LinearAlgebra.diag(Xdi[scen][load_id]*Td)
                    pd[scen][load_id] = LinearAlgebra.diag(Td*Xdr[scen][load_id])
                    qd[scen][load_id] = LinearAlgebra.diag(Td*Xdi[scen][load_id])

                    for (idx, c) in enumerate(load["connections"])
                        if abs(pd0[idx]+im*qd0[idx]) == 0.0
                            JuMP.@constraint(model, Xdr[scen][load_id][:,idx] .== 0)
                            JuMP.@constraint(model, Xdi[scen][load_id][:,idx] .== 0)
                        end
                    end

                    if load["model"]==PMD.POWER
                        for (idx, c) in enumerate(load["connections"])
                            JuMP.@constraint(model, pd[scen][load_id][idx]==pd0[idx])
                            JuMP.@constraint(model, qd[scen][load_id][idx]==qd0[idx])
                        end
                    elseif load["model"]==PMD.IMPEDANCE
                        for (idx,c) in enumerate(load["connections"])
                            JuMP.@constraint(model, pd[scen][load_id][idx]==3*a[idx]*w[scen][bus_id][[c for c in load["connections"]]][idx])
                            JuMP.@constraint(model, qd[scen][load_id][idx]==3*b[idx]*w[scen][bus_id][[c for c in load["connections"]]][idx])
                        end
                    else
                        for (idx,c) in enumerate(load["connections"])
                            JuMP.@constraint(model, pd[scen][load_id][idx]==sqrt(3)/2*a[idx]*(w[scen][bus_id][[c for c in load["connections"]]][idx]+1))
                            JuMP.@constraint(model, qd[scen][load_id][idx]==sqrt(3)/2*b[idx]*(w[scen][bus_id][[c for c in load["connections"]]][idx]+1))
                        end
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

            pd_zblock = Dict(scen => Dict(l => JuMP.@variable(model, [c in conns], base_name="0_pd_zblock_$(l)_$(scen)") for (l,conns) in ref[:bus_conns_load][i]) for scen in scenarios)
            qd_zblock = Dict(scen => Dict(l => JuMP.@variable(model, [c in conns], base_name="0_qd_zblock_$(l)_$(scen)") for (l,conns) in ref[:bus_conns_load][i]) for scen in scenarios)

            for (l,conns) in ref[:bus_conns_load][i]
                for c in conns
                    for scen in scenarios
                        IM.relaxation_product(model, pd_bus[scen][l][c], z_block[scen][ref[:load_block_map][l]], pd_zblock[scen][l][c])
                        IM.relaxation_product(model, qd_bus[scen][l][c], z_block[scen][ref[:load_block_map][l]], qd_zblock[scen][l][c])
                    end
                end
            end

            for (idx, t) in ungrounded_terminals
                for scen in scenarios
                    JuMP.@constraint(model,
                        sum(p[scen][a][t] for (a, conns) in ref[:bus_arcs_conns_branch][i] if t in conns)
                        + sum(psw[scen][a_sw][t] for (a_sw, conns) in ref[:bus_arcs_conns_switch][i] if t in conns)
                        + sum(pt[scen][a_trans][t] for (a_trans, conns) in ref[:bus_arcs_conns_transformer][i] if t in conns)
                        ==
                        sum(pg[scen][g][t] for (g, conns) in ref[:bus_conns_gen][i] if t in conns)
                        - sum(ps[scen][s][t] for (s, conns) in ref[:bus_conns_storage][i] if t in conns)
                        - sum(pd_zblock[scen][l][t] for (l, conns) in ref[:bus_conns_load][i] if t in conns)
                        - sum((w[scen][i][t] * LinearAlgebra.diag(Gt')[idx]) for (sh, conns) in ref[:bus_conns_shunt][i] if t in conns)
                    )

                    JuMP.@constraint(model,
                        sum(q[scen][a][t] for (a, conns) in ref[:bus_arcs_conns_branch][i] if t in conns)
                        + sum(qsw[scen][a_sw][t] for (a_sw, conns) in ref[:bus_arcs_conns_switch][i] if t in conns)
                        + sum(qt[scen][a_trans][t] for (a_trans, conns) in ref[:bus_arcs_conns_transformer][i] if t in conns)
                        ==
                        sum(qg[scen][g][t] for (g, conns) in ref[:bus_conns_gen][i] if t in conns)
                        - sum(qs[scen][s][t] for (s, conns) in ref[:bus_conns_storage][i] if t in conns)
                        - sum(qd_zblock[scen][l][t] for (l, conns) in ref[:bus_conns_load][i] if t in conns)
                        - sum((-w[scen][i][t] * LinearAlgebra.diag(Bt')[idx]) for (sh, conns) in uncontrolled_shunts if t in conns)
                        - sum(-qc[scen][sh][t] for (sh, conns) in controlled_shunts if t in conns)
                    )
                end

                for (sh, sh_conns) in controlled_shunts
                    if t in sh_conns
                        bs = LinearAlgebra.diag(ref[:shunt][sh]["bs"])[findfirst(isequal(t), sh_conns)]
                        for scen in scenarios
                            w_lb, w_ub = IM.variable_domain(w[scen][i][t])

                            JuMP.@constraint(model, z_cap[scen][sh] <= z_block[scen][ref[:bus_block_map][i]])
                            JuMP.@constraint(model, qc[scen][sh] ≥ bs*z_cap[scen][sh]*w_lb)
                            JuMP.@constraint(model, qc[scen][sh] ≥ bs*w[scen][t] + bs*z_cap[scen][sh]*w_ub - bs*w_ub*z_block[scen][ref[:bus_block_map][i]])
                            JuMP.@constraint(model, qc[scen][sh] ≤ bs*z_cap[scen][sh]*w_ub)
                            JuMP.@constraint(model, qc[scen][sh] ≤ bs*w[scen][t] + bs*z_cap[scen][sh]*w_lb - bs*w_lb*z_block[scen][ref[:bus_block_map][i]])
                        end
                    end
                end
            end
        end

        # storage constraints
        for (i,strg) in ref[:storage]
            for scen in scenarios
                # constraint_storage_state
                JuMP.@constraint(model, se[scen][i] - strg["energy"] == ref[:time_elapsed]*(strg["charge_efficiency"]*sc[scen][i] - sd[scen][i]/strg["discharge_efficiency"]))

                # constraint_storage_complementarity_mi_block_on_off
                JuMP.@constraint(model, sc_on[scen][i] + sd_on[scen][i] == z_block[scen][ref[:storage_block_map][i]])
                JuMP.@constraint(model, sc_on[scen][i]*strg["charge_rating"] >= sc[scen][i])
                JuMP.@constraint(model, sd_on[scen][i]*strg["discharge_rating"] >= sd[scen][i])
            end

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

            unbalance_factor = get(strg, "phase_unbalance_factor", Inf)
            for scen in scenarios
                isfinite(pmin) && JuMP.@constraint(model, sum(ps[scen][i]) >= z_block[scen][ref[:storage_block_map][i]]*pmin)
                isfinite(qmin) && JuMP.@constraint(model, sum(qs[scen][i]) >= z_block[scen][ref[:storage_block_map][i]]*qmin)

                isfinite(pmax) && JuMP.@constraint(model, sum(ps[scen][i]) <= z_block[scen][ref[:storage_block_map][i]]*pmax)
                isfinite(qmax) && JuMP.@constraint(model, sum(qs[scen][i]) <= z_block[scen][ref[:storage_block_map][i]]*qmax)

                # constraint_mc_storage_losses_block_on_off
                if JuMP.has_lower_bound(qsc[scen][i]) && JuMP.has_upper_bound(qsc[scen][i])
                    qsc_zblock = JuMP.@variable(model, base_name="0_qd_zblock_$(i)_$(scen)")

                    JuMP.@constraint(model, qsc_zblock >= JuMP.lower_bound(qsc[scen][i]) * z_block[scen][ref[:storage_block_map][i]])
                    JuMP.@constraint(model, qsc_zblock >= JuMP.upper_bound(qsc[scen][i]) * z_block[scen][ref[:storage_block_map][i]] + qsc[scen][i] - JuMP.upper_bound(qsc[scen][i]))
                    JuMP.@constraint(model, qsc_zblock <= JuMP.upper_bound(qsc[scen][i]) * z_block[scen][ref[:storage_block_map][i]])
                    JuMP.@constraint(model, qsc_zblock <= qsc[scen][i] + JuMP.lower_bound(qsc[scen][i]) * z_block[scen][ref[:storage_block_map][i]] - JuMP.lower_bound(qsc[scen][i]))

                    JuMP.@constraint(model, sum(qs[scen][i]) == qsc_zblock + strg["q_loss"] * z_block[scen][ref[:storage_block_map][i]])
                else
                    # Note that this is not supported in LP solvers when z_block is continuous
                    JuMP.@constraint(model, sum(qs[scen][i]) == qsc[scen][i] * z_block[scen][ref[:storage_block_map][i]] + strg["q_loss"] * z_block[scen][ref[:storage_block_map][i]])
                end
                JuMP.@constraint(model, sum(ps[scen][i]) + (sd[scen][i] - sc[scen][i]) == strg["p_loss"] * z_block[scen][ref[:storage_block_map][i]])

                # constraint_mc_storage_thermal_limit
                _ps = [ps[scen][i][c] for c in strg["connections"]]
                _qs = [qs[scen][i][c] for c in strg["connections"]]

                ps_sqr = [JuMP.@variable(model, base_name="0_ps_sqr_$(i)_$(c)_$(scen)") for c in strg["connections"]]
                qs_sqr = [JuMP.@variable(model, base_name="0_qs_sqr_$(i)_$(c)_$(scen)") for c in strg["connections"]]

                for (idx,c) in enumerate(strg["connections"])
                    ps_lb, ps_ub = IM.variable_domain(_ps[idx])
                    PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, _ps[idx], ps_sqr[idx], [ps_lb, ps_ub], false)

                    qs_lb, qs_ub = IM.variable_domain(_qs[idx])
                    PMD.PolyhedralRelaxations.construct_univariate_relaxation!(model, x->x^2, _qs[idx], qs_sqr[idx], [qs_lb, qs_ub], false)
                end

                JuMP.@constraint(model, sum(ps_sqr .+ qs_sqr) <= strg["thermal_rating"]^2)
            end

            # constraint_mc_storage_phase_unbalance_grid_following
            if isfinite(unbalance_factor)
                for scen in scenarios
                    sd_on_ps = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_ps_$(i)_$(scen)")
                    sc_on_ps = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_ps_$(i)_$(scen)")
                    sd_on_qs = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_qs_$(i)_$(scen)")
                    sc_on_qs = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_qs_$(i)_$(scen)")
                    ps_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_ps_zinverter_$(i)_$(scen)")
                    qs_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_qs_zinverter_$(i)_$(scen)")
                    sd_on_ps_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_ps_zinverter_$(i)_$(scen)")
                    sc_on_ps_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_ps_zinverter_$(i)_$(scen)")
                    sd_on_qs_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sd_on_qs_zinverter_$(i)_$(scen)")
                    sc_on_qs_zinverter = JuMP.@variable(model, [c in strg["connections"]], base_name="0_sc_on_qs_zinverter_$(i)_$(scen)")
                    for c in strg["connections"]
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sd_on[scen][i], ps[scen][i][c], sd_on_ps[c], [0,1], [JuMP.lower_bound(ps[scen][i][c]), JuMP.upper_bound(ps[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sc_on[scen][i], ps[scen][i][c], sc_on_ps[c], [0,1], [JuMP.lower_bound(ps[scen][i][c]), JuMP.upper_bound(ps[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sd_on[scen][i], qs[scen][i][c], sd_on_qs[c], [0,1], [JuMP.lower_bound(qs[scen][i][c]), JuMP.upper_bound(qs[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, sc_on[scen][i], qs[scen][i][c], sc_on_qs[c], [0,1], [JuMP.lower_bound(qs[scen][i][c]), JuMP.upper_bound(qs[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], ps[scen][i][c], ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[scen][i][c]), JuMP.upper_bound(ps[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], qs[scen][i][c], qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[scen][i][c]), JuMP.upper_bound(qs[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], sd_on_ps[scen][c], sd_on_ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[scen][i][c]), JuMP.upper_bound(ps[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], sc_on_ps[scen][c], sc_on_ps_zinverter[c], [0,1], [JuMP.lower_bound(ps[scen][i][c]), JuMP.upper_bound(ps[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], sd_on_qs[scen][c], sd_on_qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[scen][i][c]), JuMP.upper_bound(qs[scen][i][c])])
                        PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(model, z_inverter[scen][(:storage,i)], sc_on_qs[scen][c], sc_on_qs_zinverter[c], [0,1], [JuMP.lower_bound(qs[scen][i][c]), JuMP.upper_bound(qs[scen][i][c])])
                    end
                end

                for (idx,c) in enumerate(strg["connections"])
                    if idx < length(strg["connections"])
                        for d in strg["connections"][idx+1:end]
                            for scen in scenarios
                                JuMP.@constraint(model, ps[scen][i][c]-ps_zinverter[scen][c] >= ps[scen][i][d] - unbalance_factor*(-1*sd_on_ps[scen][d] + 1*sc_on_ps[scen][d]) - ps_zinverter[scen][d] + unbalance_factor*(-1*sd_on_ps_zinverter[scen][d] + 1*sc_on_ps_zinverter[scen][d]))
                                JuMP.@constraint(model, ps[scen][i][c]-ps_zinverter[scen][c] <= ps[scen][i][d] + unbalance_factor*(-1*sd_on_ps[scen][d] + 1*sc_on_ps[scen][d]) - ps_zinverter[scen][d] - unbalance_factor*(-1*sd_on_ps_zinverter[scen][d] + 1*sc_on_ps_zinverter[scen][d]))

                                JuMP.@constraint(model, qs[scen][i][c]-qs_zinverter[scen][c] >= qs[scen][i][d] - unbalance_factor*(-1*sd_on_qs[scen][d] + 1*sc_on_qs[scen][d]) - qs_zinverter[scen][d] + unbalance_factor*(-1*sd_on_qs_zinverter[scen][d] + 1*sc_on_qs_zinverter[scen][d]))
                                JuMP.@constraint(model, qs[scen][i][c]-qs_zinverter[scen][c] <= qs[scen][i][d] + unbalance_factor*(-1*sd_on_qs[scen][d] + 1*sc_on_qs[scen][d]) - qs_zinverter[scen][d] - unbalance_factor*(-1*sd_on_qs_zinverter[scen][d] + 1*sc_on_qs_zinverter[scen][d]))
                            end
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

            for scen in scenarios
                p_fr = p[scen][f_idx]
                q_fr = q[scen][f_idx]

                p_to = p[scen][t_idx]
                q_to = q[scen][t_idx]

                w_fr = w[scen][f_bus]
                w_to = w[scen][t_bus]

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

                for scen in scenarios
                    w_fr = w[scen][f_bus][fc]
                    p_fr = p[scen][f_idx][fc]
                    q_fr = q[scen][f_idx][fc]

                    JuMP.@constraint(model,
                        tan(angmin[idx])*((1 + r*g_fr - x*b_fr)*(w_fr) - r*p_fr - x*q_fr)
                                <= ((-x*g_fr - r*b_fr)*(w_fr) + x*p_fr - r*q_fr)
                        )
                    JuMP.@constraint(model,
                        tan(angmax[idx])*((1 + r*g_fr - x*b_fr)*(w_fr) - r*p_fr - x*q_fr)
                                >= ((-x*g_fr - r*b_fr)*(w_fr) + x*p_fr - r*q_fr)
                        )
                end
            end

            # ampacity constraints
            if haskey(branch, "c_rating_a") && any(branch["c_rating_a"] .< Inf)
                c_rating = branch["c_rating_a"]

                for scen in scenarios

                    # constraint_mc_ampacity_from
                    p_fr = [p[scen][f_idx][c] for c in f_connections]
                    q_fr = [q[scen][f_idx][c] for c in f_connections]
                    w_fr = [w[scen][f_idx[2]][c] for c in f_connections]

                    p_sqr_fr = [JuMP.@variable(model, base_name="0_p_sqr_fr_$(f_idx)[$(c)]_$(scen)") for c in f_connections]
                    q_sqr_fr = [JuMP.@variable(model, base_name="0_q_sqr_fr_$(f_idx)[$(c)]_$(scen)") for c in f_connections]

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
                    p_to = [p[scen][t_idx][c] for c in t_connections]
                    q_to = [q[scen][t_idx][c] for c in t_connections]
                    w_to = [w[scen][t_idx[2]][c] for c in t_connections]

                    p_sqr_to = [JuMP.@variable(model, base_name="0_p_sqr_to_$(t_idx)[$(c)]_$(scen)") for c in t_connections]
                    q_sqr_to = [JuMP.@variable(model, base_name="0_q_sqr_to_$(t_idx)[$(c)]_$(scen)") for c in t_connections]

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
        end

        # constraint_switch_close_action_limit
        if switch_close_actions_ub < Inf
            Δᵞs = Dict(scen => Dict(l => JuMP.@variable(model, base_name="0_delta_switch_state_$(l)_$(scen)") for l in keys(ref[:switch_dispatchable])) for scen in scenarios)
            for scen in scenarios
                for (s, Δᵞ) in Δᵞs[scen]
                    γ = z_switch[scen][s]
                    γ₀ = JuMP.start_value(γ)
                    JuMP.@constraint(model, Δᵞ >=  γ * (1 - γ₀))
                    JuMP.@constraint(model, Δᵞ >= -γ * (1 - γ₀))
                end
                JuMP.@constraint(model, sum(Δᵞ for (l, Δᵞ) in Δᵞs[scen]) <= switch_close_actions_ub)
            end
        end

        # constraint_radial_topology
        f_rad = Dict(scen => Dict() for scen in scenarios)
        λ = Dict(scen => Dict() for scen in scenarios)
        β = Dict(scen => Dict() for scen in scenarios)
        α = Dict(scen => Dict() for scen in scenarios)

        for (s,sw) in ref[:switch]
            (i,j) = (ref[:bus_block_map][sw["f_bus"]], ref[:bus_block_map][sw["t_bus"]])
            for scen in scenarios
                α[scen][(i,j)] = z_switch[scen][s]
            end
        end

        for scen in scenarios
            for (i,j) in _L′
                for k in filter(kk->kk∉iᵣ,_N)
                    f_rad[scen][(k, i, j)] = JuMP.@variable(model, base_name="0_f_$((k,i,j))_$(scen)")
                end
                λ[scen][(i,j)] = JuMP.@variable(model, base_name="0_lambda_$((i,j))_$(scen)", binary=true, lower_bound=0, upper_bound=1)

                if (i,j) ∈ _L₀
                    β[scen][(i,j)] = JuMP.@variable(model, base_name="0_beta_$((i,j))_$(scen)", lower_bound=0, upper_bound=1)
                end
            end

            JuMP.@constraint(model, sum((λ[scen][(i,j)] + λ[scen][(j,i)]) for (i,j) in _L) == length(_N) - 1)

            for (i,j) in _L₀
                JuMP.@constraint(model, λ[scen][(i,j)] + λ[scen][(j,i)] == β[scen][(i,j)])
                JuMP.@constraint(model, α[scen][(i,j)] <= β[scen][(i,j)])
            end
        end

        for k in filter(kk->kk∉iᵣ,_N)
            for _iᵣ in iᵣ
                jiᵣ = filter(((j,i),)->i==_iᵣ&&i!=j,_L)
                iᵣj = filter(((i,j),)->i==_iᵣ&&i!=j,_L)
                if !(isempty(jiᵣ) && isempty(iᵣj))
                    for scen in scenarios
                        JuMP.@constraint(
                            model,
                            sum(f_rad[scen][(k,j,i)] for (j,i) in jiᵣ) -
                            sum(f_rad[scen][(k,i,j)] for (i,j) in iᵣj)
                            ==
                            -1.0
                        )
                    end
                end
            end

            jk = filter(((j,i),)->i==k&&i!=j,_L′)
            kj = filter(((i,j),)->i==k&&i!=j,_L′)
            if !(isempty(jk) && isempty(kj))
                for scen in scenarios
                    JuMP.@constraint(
                        model,
                        sum(f_rad[scen][(k,j,k)] for (j,i) in jk) -
                        sum(f_rad[scen][(k,k,j)] for (i,j) in kj)
                        ==
                        1.0
                    )
                end
            end

            for i in filter(kk->kk∉iᵣ&&kk!=k,_N)
                ji = filter(((j,ii),)->ii==i&&ii!=j,_L′)
                ij = filter(((ii,j),)->ii==i&&ii!=j,_L′)
                if !(isempty(ji) && isempty(ij))
                    for scen in scenarios
                        JuMP.@constraint(
                            model,
                            sum(f_rad[scen][(k,j,i)] for (j,ii) in ji) -
                            sum(f_rad[scen][(k,i,j)] for (ii,j) in ij)
                            ==
                            0.0
                        )
                    end
                end
            end

            for (i,j) in _L
                for scen in scenarios
                    JuMP.@constraint(model, f_rad[scen][(k,i,j)] >= 0)
                    JuMP.@constraint(model, f_rad[scen][(k,i,j)] <= λ[scen][(i,j)])
                    JuMP.@constraint(model, f_rad[scen][(k,j,i)] >= 0)
                    JuMP.@constraint(model, f_rad[scen][(k,j,i)] <= λ[scen][(j,i)])
                end
            end
        end

        # constraint_isolate_block
        for (s, switch) in ref[:switch_dispatchable]
            for scen in scenarios
                z_block_fr = z_block[scen][ref[:bus_block_map][switch["f_bus"]]]
                z_block_to = z_block[scen][ref[:bus_block_map][switch["t_bus"]]]

                γ = z_switch[scen][s]
                JuMP.@constraint(model,  (z_block_fr - z_block_to) <=  (1-γ))
                JuMP.@constraint(model,  (z_block_fr - z_block_to) >= -(1-γ))
            end
        end

        for b in keys(ref[:blocks])
            n_gen = length(ref[:block_gens][b])
            n_strg = length(ref[:block_storages][b])
            n_neg_loads = length([_b for (_b,ls) in ref[:block_loads] if any(any(ref[:load][l]["pd"] .< 0) for l in ls)])
            for scen in scenarios
                JuMP.@constraint(model, z_block[scen][b] <= n_gen + n_strg + n_neg_loads + sum(z_switch[scen][s] for s in keys(ref[:block_switches]) if s in keys(ref[:switch_dispatchable])))
            end
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

            for scen in scenarios
                w_fr = w[scen][f_bus_id]
                w_to = w[scen][f_bus_id]

                # constraint_mc_switch_state_open_close
                for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
                    JuMP.@constraint(model, w_fr[fc] - w_to[tc] <=  vmax[idx].^2 * (1-z_switch[scen][i]))
                    JuMP.@constraint(model, w_fr[fc] - w_to[tc] >= -vmax[idx].^2 * (1-z_switch[scen][i]))
                end

                for (idx, c) in enumerate(f_connections)
                    JuMP.@constraint(model, psw[scen][f_idx][c] <=  rating[idx] * z_switch[scen][i])
                    JuMP.@constraint(model, psw[scen][f_idx][c] >= -rating[idx] * z_switch[scen][i])
                    JuMP.@constraint(model, qsw[scen][f_idx][c] <=  rating[idx] * z_switch[scen][i])
                    JuMP.@constraint(model, qsw[scen][f_idx][c] >= -rating[idx] * z_switch[scen][i])
                end
            end

            # constraint_mc_switch_ampacity
            if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf)
                c_rating = switch["current_rating"]

                for scen in scenarios
                    psw_fr = [psw[scen][f_idx][c] for c in f_connections]
                    qsw_fr = [qsw[scen][f_idx][c] for c in f_connections]
                    w_fr = [w[scen][f_idx[2]][c] for c in f_connections]

                    psw_sqr_fr = [JuMP.@variable(model, base_name="0_psw_sqr_$(f_idx)[$(c)]_$(scen)") for c in f_connections]
                    qsw_sqr_fr = [JuMP.@variable(model, base_name="0_qsw_sqr_$(f_idx)[$(c)]_$(scen)") for c in f_connections]

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
        end

        # transformer constraints
        for (i,transformer) in ref[:transformer]
            f_bus = transformer["f_bus"]
            t_bus = transformer["t_bus"]
            f_idx = (i, f_bus, t_bus)
            t_idx = (i, t_bus, f_bus)
            configuration = transformer["configuration"]
            f_connections = transformer["f_connections"]
            t_connections = transformer["t_connections"]
            tm_set = transformer["tm_set"]
            tm_fixed = transformer["tm_fix"]
            tm_scale = PMD.calculate_tm_scale(transformer, ref[:bus][f_bus], ref[:bus][t_bus])
            pol = transformer["polarity"]

            if configuration == PMD.WYE
                for scen in scenarios
                    tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[scen][idx] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]

                    p_fr = [pt[scen][f_idx][p] for p in f_connections]
                    p_to = [pt[scen][t_idx][p] for p in t_connections]
                    q_fr = [qt[scen][f_idx][p] for p in f_connections]
                    q_to = [qt[scen][t_idx][p] for p in t_connections]

                    w_fr = w[scen][f_bus]
                    w_to = w[scen][t_bus]

                    tmsqr = [
                        tm_fixed[i] ? tm[i]^2 : JuMP.@variable(
                            model,
                            base_name="0_tmsqr_$(trans_id)_$(f_connections[i])_$(scen)",
                            start=JuMP.start_value(tm[i])^2,
                            lower_bound=JuMP.has_lower_bound(tm[i]) ? JuMP.lower_bound(tm[i])^2 : 0.9^2,
                            upper_bound=JuMP.has_upper_bound(tm[i]) ? JuMP.upper_bound(tm[i])^2 : 1.1^2
                        ) for i in 1:length(tm)
                    ]

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

                            tmsqr_w_to = JuMP.@variable(model, base_name="0_tmsqr_w_to_$(trans_id)_$(t_bus)_$(tc)_$(scen)")
                            PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(
                                model,
                                tmsqr[scen][idx],
                                w_to[tc],
                                tmsqr_w_to,
                                [JuMP.lower_bound(tmsqr[scen][idx]), JuMP.upper_bound(tmsqr[scen][idx])],
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
                end

            elseif configuration == PMD.DELTA
                for scen in scenarios
                    tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[scen][fc] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]
                    nph = length(tm_set)

                    p_fr = [pt[scen][f_idx][p] for p in f_connections]
                    p_to = [pt[scen][t_idx][p] for p in t_connections]
                    q_fr = [qt[scen][f_idx][p] for p in f_connections]
                    q_to = [qt[scen][t_idx][p] for p in t_connections]

                    w_fr = w[scen][f_bus]
                    w_to = w[scen][t_bus]

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
        end

        # objective
        delta_sw_state = Dict(scen => JuMP.@variable(
            model,
            [i in keys(ref[:switch_dispatchable])],
            base_name="$(i)_delta_sw_state",
        ) for scen in scenarios)

        for (s,switch) in ref[:switch_dispatchable]
            for scen in scenarios
                JuMP.@constraint(model, delta_sw_state[scen][s] >=  (switch["state"] - z_switch[scen][s]))
                JuMP.@constraint(model, delta_sw_state[scen][s] >= -(switch["state"] - z_switch[scen][s]))
            end
        end

        for i in keys(ref[:switch_dispatchable])
            JuMP.@constraint(model, z_switch[1][i] == z_switch[2][i])
        end

        for t in [:storage, :gen]
            for i in keys(ref[t])
                if get(ref[t][i], "inverter", 1) == 1
                    JuMP.@constraint(model, z_inverter[1][(t,i)] == z_inverter[2][(t,i)])
                end
            end
        end


        JuMP.@objective(model, Min, sum(
                sum( block_weights[i] * (1-z_block[scen][i]) for (i,block) in ref[:blocks])
                - obj_fac[scen]*sum( ref[:switch_scores][l]*(1-z_switch[scen][l]) for l in keys(ref[:switch_dispatchable]) )
                + sum( delta_sw_state[scen][l] for l in keys(ref[:switch_dispatchable])) / n_dispatchable_switches
                + sum( (strg["energy_rating"] - se[scen][i]) for (i,strg) in ref[:storage]) / total_energy_ub
                + sum( sum(get(gen,  "cost", [0.0, 0.0])[2] * pg[scen][i][c] + get(gen,  "cost", [0.0, 0.0])[1] for c in  gen["connections"]) for (i,gen) in ref[:gen]) / total_energy_ub
        for scen in scenarios) )

        ## solve manual model
        JuMP.optimize!(model)
        sts = string(JuMP.termination_status(model))
        println("$(scenarios): $(sts)")
        for scen in scenarios
        println("$(scen) switch status: $([JuMP.value(z_switch[scen][i]) for i in keys(ref[:switch_dispatchable])])")
        println("$(scen) inverter status: $([JuMP.value(z_inv[i]) for ((t,i), z_inv) in z_inverter[scen]])")
        println("$(scen) block status: $([JuMP.value(z_block[scen][i]) for i in keys(ref[:blocks])])")
        end
        # if scenario == N_scenarios
            # for scen in scenarios
            #     println("Substation pg for scenario $(scen): $([JuMP.value(pg[scen][4][c]) for c=1:3])")
            # end
        # end
        # z_inv_sol = Dict(
        #     (t,i) => JuMP.value(z_inv) for ((t,i), z_inv) in z_inverter
        # )
        # z_sw_sol = Dict(i => JuMP.value(z_switch[scen][i]) for i in keys(ref[:switch_dispatchable]))

        # run power flow to identify maximum constraint violations
        # N_viol = Dict()
        # for scen in deleteat!([1:N_scenarios;], sort(scenarios))
        #     N_viol[scen] = solve_pf(z_inv_sol,z_sw_sol,load_factor[scen])
        #     println("$(scen): $(N_viol)")
        # end
        # if isempty(N_viol) || findmax(N_viol)[1]==0
        #     viol_ind = false
        # elseif !isempty(N_viol) && findmax(N_viol)[1]!=0
        #     push!(scenarios,findmax(N_viol)[2])
        # end

        # feasibility check
        # scenario = deleteat!([1:N_scenarios;], sort(scenarios))
        # scen = 1
        # while !isempty(scenario)
        #     sts = feasibility_check(z_inv_sol,z_sw_sol,load_factor[scenario[scen]])
        #     println("$(scenario[scen]) $sts")
        #     if sts=="OPTIMAL" && scen!=length(scenario)
        #         scen += 1
        #     elseif sts!="OPTIMAL"
        #         push!(scenarios,scenario[scen])
        #         scenario = []
        #     elseif sts=="OPTIMAL" && scen==length(scenario)
        #         viol_ind = false
        #         scenario = []
        #     end
        # end
        viol_ind = false
    end
end
