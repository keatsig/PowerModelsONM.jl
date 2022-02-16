@doc raw"""
    constraint_mc_switch_voltage_open_close(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, f_bus::Int, t_bus::Int, f_connections::Vector{Int}, t_connections::Vector{Int})

Linear switch power on/off constraint for LPUBFDiagModel.

```math
\begin{align}
& w^{fr}_{i,c} - w^{to}_{i,c} \leq \left ( v^u_{i,c} \right )^2 \left ( 1 - z^{sw}_i \right )\ \forall i \in S,\forall c \in C \\
& w^{fr}_{i,c} - w^{to}_{i,c} \geq -\left ( v^u_{i,c}\right )^2 \left ( 1 - z^{sw}_i \right )\ \forall i \in S,\forall c \in C
\end{align}
```
"""
function constraint_mc_switch_voltage_open_close(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, f_bus::Int, t_bus::Int, f_connections::Vector{Int}, t_connections::Vector{Int})
    w_fr = var(pm, nw, :w, f_bus)
    w_to = var(pm, nw, :w, t_bus)

    f_bus = ref(pm, nw, :bus, f_bus)
    t_bus = ref(pm, nw, :bus, t_bus)

    f_vmax = f_bus["vmax"][[findfirst(isequal(c), f_bus["terminals"]) for c in f_connections]]
    t_vmax = t_bus["vmax"][[findfirst(isequal(c), t_bus["terminals"]) for c in t_connections]]

    vmax = min.(fill(2.0, length(f_bus["vmax"])), f_vmax, t_vmax)

    z = var(pm, nw, :switch_state, i)

    for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
        JuMP.@constraint(pm.model, w_fr[fc] - w_to[tc] <=  vmax[idx].^2 * (1-z))
        JuMP.@constraint(pm.model, w_fr[fc] - w_to[tc] >= -vmax[idx].^2 * (1-z))

        # Indicator constraint version, for reference
        # JuMP.@constraint(pm.model, z => {w_fr[fc] == w_to[tc]})
    end
end


"""
    constraint_mc_power_balance_shed_block(pm::PMD.LPUBFDiagModel, nw::Int, i::Int,
        terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}},
        bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}},
        bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}},
        bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}}
    )

KCL for block load shed problem with transformers (LinDistFlow Form)
"""
function constraint_mc_power_balance_shed_block(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}}, bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}})
    w        = var(pm, nw, :w, i)
    p        = get(var(pm, nw),    :p, Dict()); PMD._check_var_keys(p, bus_arcs, "active power", "branch")
    q        = get(var(pm, nw),    :q, Dict()); PMD._check_var_keys(q, bus_arcs, "reactive power", "branch")
    pg       = get(var(pm, nw),   :pg, Dict()); PMD._check_var_keys(pg, bus_gens, "active power", "generator")
    qg       = get(var(pm, nw),   :qg, Dict()); PMD._check_var_keys(qg, bus_gens, "reactive power", "generator")
    ps       = get(var(pm, nw),   :ps, Dict()); PMD._check_var_keys(ps, bus_storage, "active power", "storage")
    qs       = get(var(pm, nw),   :qs, Dict()); PMD._check_var_keys(qs, bus_storage, "reactive power", "storage")
    psw      = get(var(pm, nw),  :psw, Dict()); PMD._check_var_keys(psw, bus_arcs_sw, "active power", "switch")
    qsw      = get(var(pm, nw),  :qsw, Dict()); PMD._check_var_keys(qsw, bus_arcs_sw, "reactive power", "switch")
    pt       = get(var(pm, nw),   :pt, Dict()); PMD._check_var_keys(pt, bus_arcs_trans, "active power", "transformer")
    qt       = get(var(pm, nw),   :qt, Dict()); PMD._check_var_keys(qt, bus_arcs_trans, "reactive power", "transformer")
    pd       = get(var(pm, nw), :pd_bus,  Dict()); PMD._check_var_keys(pd,  bus_loads, "active power", "load")
    qd       = get(var(pm, nw), :qd_bus,  Dict()); PMD._check_var_keys(qd,  bus_loads, "reactive power", "load")
    z_block  = var(pm, nw, :z_block, ref(pm, nw, :bus_block_map, i))

    uncontrolled_shunts = Tuple{Int,Vector{Int}}[]
    controlled_shunts = Tuple{Int,Vector{Int}}[]

    if !isempty(bus_shunts) && any(haskey(ref(pm, nw, :shunt, sh), "controls") for (sh, conns) in bus_shunts)
        for (sh, conns) in bus_shunts
            if haskey(ref(pm, nw, :shunt, sh), "controls")
                push!(controlled_shunts, (sh,conns))
            else
                push!(uncontrolled_shunts, (sh, conns))
            end
        end
    else
        uncontrolled_shunts = bus_shunts
    end

    Gt, _ = PMD._build_bus_shunt_matrices(pm, nw, terminals, bus_shunts)
    _, Bt = PMD._build_bus_shunt_matrices(pm, nw, terminals, uncontrolled_shunts)

    cstr_p = []
    cstr_q = []

    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]

    pd_zblock = Dict{Int,JuMP.Containers.DenseAxisArray{JuMP.VariableRef}}(l => JuMP.@variable(pm.model, [c in conns], base_name="$(nw)_pd_zblock_$(l)") for (l,conns) in bus_loads)
    qd_zblock = Dict{Int,JuMP.Containers.DenseAxisArray{JuMP.VariableRef}}(l => JuMP.@variable(pm.model, [c in conns], base_name="$(nw)_qd_zblock_$(l)") for (l,conns) in bus_loads)

    for (l,conns) in bus_loads
        for c in conns
            _IM.relaxation_product(pm.model, pd[l][c], z_block, pd_zblock[l][c])
            _IM.relaxation_product(pm.model, qd[l][c], z_block, qd_zblock[l][c])
        end
    end

    for (idx, t) in ungrounded_terminals
        cp = JuMP.@constraint(pm.model,
            sum(p[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(psw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(pt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(pg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(ps[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(pd_zblock[l][t] for (l, conns) in bus_loads if t in conns)
            - sum((w[t] * LinearAlgebra.diag(Gt')[idx]) for (sh, conns) in bus_shunts if t in conns)
        )
        push!(cstr_p, cp)

        for (sh, sh_conns) in controlled_shunts
            if t in sh_conns
                cq_cap = var(pm, nw, :capacitor_reactive_power, sh)[t]
                cap_state = var(pm, nw, :capacitor_state, sh)[t]
                bs = PMD.diag(ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(t), sh_conns)]
                w_lb, w_ub = _IM.variable_domain(w[t])

                # tie to z_block
                JuMP.@constraint(pm.model, cap_state <= z_block)

                # McCormick envelope constraints
                JuMP.@constraint(pm.model, cq_cap ≥ bs*cap_state*w_lb)
                JuMP.@constraint(pm.model, cq_cap ≥ bs*w[t] + bs*cap_state*w_ub - bs*w_ub*z_block)
                JuMP.@constraint(pm.model, cq_cap ≤ bs*cap_state*w_ub)
                JuMP.@constraint(pm.model, cq_cap ≤ bs*w[t] + bs*cap_state*w_lb - bs*w_lb*z_block)
            end
        end

        cq = JuMP.@constraint(pm.model,
            sum(q[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(qsw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(qt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(qg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(qs[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(qd_zblock[l][t] for (l, conns) in bus_loads if t in conns)
            - sum((-w[t] * LinearAlgebra.diag(Bt')[idx]) for (sh, conns) in uncontrolled_shunts if t in conns)
            - sum(-var(pm, nw, :capacitor_reactive_power, sh)[t] for (sh, conns) in controlled_shunts if t in conns)
        )
        push!(cstr_q, cq)
    end

    PMD.con(pm, nw, :lam_kcl_r)[i] = cstr_p
    PMD.con(pm, nw, :lam_kcl_i)[i] = cstr_q

    if _IM.report_duals(pm)
        sol(pm, nw, :bus, i)[:lam_kcl_r] = cstr_p
        sol(pm, nw, :bus, i)[:lam_kcl_i] = cstr_q
    end
end


"""
    constraint_mc_power_balance_shed_traditional(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}}, bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}})

KCL for traditional load shed problem with transformers (LinDistFlow Form)
"""
function constraint_mc_power_balance_shed_traditional(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}}, bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}})
    w        = var(pm, nw, :w, i)
    p        = get(var(pm, nw),    :p, Dict()); PMD._check_var_keys(p, bus_arcs, "active power", "branch")
    q        = get(var(pm, nw),    :q, Dict()); PMD._check_var_keys(q, bus_arcs, "reactive power", "branch")
    pg       = get(var(pm, nw),   :pg, Dict()); PMD._check_var_keys(pg, bus_gens, "active power", "generator")
    qg       = get(var(pm, nw),   :qg, Dict()); PMD._check_var_keys(qg, bus_gens, "reactive power", "generator")
    ps       = get(var(pm, nw),   :ps, Dict()); PMD._check_var_keys(ps, bus_storage, "active power", "storage")
    qs       = get(var(pm, nw),   :qs, Dict()); PMD._check_var_keys(qs, bus_storage, "reactive power", "storage")
    psw      = get(var(pm, nw),  :psw, Dict()); PMD._check_var_keys(psw, bus_arcs_sw, "active power", "switch")
    qsw      = get(var(pm, nw),  :qsw, Dict()); PMD._check_var_keys(qsw, bus_arcs_sw, "reactive power", "switch")
    pt       = get(var(pm, nw),   :pt, Dict()); PMD._check_var_keys(pt, bus_arcs_trans, "active power", "transformer")
    qt       = get(var(pm, nw),   :qt, Dict()); PMD._check_var_keys(qt, bus_arcs_trans, "reactive power", "transformer")
    pd       = get(var(pm, nw), :pd_bus,  Dict()); PMD._check_var_keys(pd,  bus_loads, "active power", "load")
    qd       = get(var(pm, nw), :qd_bus,  Dict()); PMD._check_var_keys(qd,  bus_loads, "reactive power", "load")

    z_voltage= var(pm, nw, :z_voltage, i)
    z_demand = Dict(l => var(pm, nw, :z_demand, l) for (l,_) in bus_loads)

    uncontrolled_shunts = Tuple{Int,Vector{Int}}[]
    controlled_shunts = Tuple{Int,Vector{Int}}[]

    if !isempty(bus_shunts) && any(haskey(ref(pm, nw, :shunt, sh), "controls") for (sh, conns) in bus_shunts)
        for (sh, conns) in bus_shunts
            if haskey(ref(pm, nw, :shunt, sh), "controls")
                push!(controlled_shunts, (sh,conns))
            else
                push!(uncontrolled_shunts, (sh, conns))
            end
        end
    else
        uncontrolled_shunts = bus_shunts
    end

    Gt, _ = PMD._build_bus_shunt_matrices(pm, nw, terminals, bus_shunts)
    _, Bt = PMD._build_bus_shunt_matrices(pm, nw, terminals, uncontrolled_shunts)

    cstr_p = []
    cstr_q = []

    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]

    pd_zdemand = Dict{Int,JuMP.Containers.DenseAxisArray{JuMP.VariableRef}}(l => JuMP.@variable(pm.model, [c in conns], base_name="$(nw)_pd_zdemand_$(l)") for (l,conns) in bus_loads)
    qd_zdemand = Dict{Int,JuMP.Containers.DenseAxisArray{JuMP.VariableRef}}(l => JuMP.@variable(pm.model, [c in conns], base_name="$(nw)_qd_zdemand_$(l)") for (l,conns) in bus_loads)

    for (l,conns) in bus_loads
        for c in conns
            _IM.relaxation_product(pm.model, pd[l][c], z_demand[l], pd_zdemand[l][c])
            _IM.relaxation_product(pm.model, qd[l][c], z_demand[l], qd_zdemand[l][c])
        end
    end

    for (idx, t) in ungrounded_terminals
        cp = JuMP.@constraint(pm.model,
            sum(p[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(psw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(pt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(pg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(ps[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(pd_zdemand[l][t] for (l, conns) in bus_loads if t in conns)
            - sum((w[t] * LinearAlgebra.diag(Gt')[idx]) for (sh, conns) in bus_shunts if t in conns)
        )
        push!(cstr_p, cp)

        for (sh, sh_conns) in controlled_shunts
            if t in sh_conns
                cq_cap = var(pm, nw, :capacitor_reactive_power, sh)[t]
                cap_state = var(pm, nw, :capacitor_state, sh)[t]
                bs = PMD.diag(ref(pm, nw, :shunt, sh, "bs"))[findfirst(isequal(t), sh_conns)]
                w_lb, w_ub = _IM.variable_domain(w[t])

                # tie to z_voltage
                JuMP.@constraint(pm.model, cap_state <= z_voltage)

                # McCormick envelope constraints
                JuMP.@constraint(pm.model, cq_cap ≥ bs*cap_state*w_lb)
                JuMP.@constraint(pm.model, cq_cap ≥ bs*w[t] + bs*cap_state*w_ub - bs*w_ub*z_voltage)
                JuMP.@constraint(pm.model, cq_cap ≤ bs*cap_state*w_ub)
                JuMP.@constraint(pm.model, cq_cap ≤ bs*w[t] + bs*cap_state*w_lb - bs*w_lb*z_voltage)
            end
        end

        cq = JuMP.@constraint(pm.model,
            sum(q[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(qsw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(qt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(qg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(qs[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(qd_zdemand[l][t] for (l, conns) in bus_loads if t in conns)
            - sum((-w[t] * LinearAlgebra.diag(Bt')[idx]) for (sh, conns) in uncontrolled_shunts if t in conns)
            - sum(-var(pm, nw, :capacitor_reactive_power, sh)[t] for (sh, conns) in controlled_shunts if t in conns)
        )
        push!(cstr_q, cq)
    end

    PMD.con(pm, nw, :lam_kcl_r)[i] = cstr_p
    PMD.con(pm, nw, :lam_kcl_i)[i] = cstr_q

    if _IM.report_duals(pm)
        sol(pm, nw, :bus, i)[:lam_kcl_r] = cstr_p
        sol(pm, nw, :bus, i)[:lam_kcl_i] = cstr_q
    end
end


@doc raw"""
    constraint_mc_transformer_power_yy_block_on_off(
        pm::PMD.LPUBFDiagModel,
        nw::Int,
        trans_id::Int,
        f_bus::Int,
        t_bus::Int,
        f_idx::Tuple{Int,Int,Int},
        t_idx::Tuple{Int,Int,Int},
        f_connections::Vector{Int},
        t_connections::Vector{Int},
        pol::Int,
        tm_set::Vector{<:Real},
        tm_fixed::Vector{Bool},
        tm_scale::Real
    )

Links to and from power and voltages in a wye-wye transformer, assumes tm_fixed is true

```math
w_fr_i=(pol_i*tm_scale*tm_i)^2w_to_i
```
"""
function constraint_mc_transformer_power_yy_block_on_off(pm::PMD.LPUBFDiagModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    z_block = var(pm, nw, :z_block, ref(pm, nw, :bus_block_map, f_bus))

    tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[idx] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]
    transformer = ref(pm, nw, :transformer, trans_id)

    p_fr = [var(pm, nw, :pt, f_idx)[p] for p in f_connections]
    p_to = [var(pm, nw, :pt, t_idx)[p] for p in t_connections]
    q_fr = [var(pm, nw, :qt, f_idx)[p] for p in f_connections]
    q_to = [var(pm, nw, :qt, t_idx)[p] for p in t_connections]

    w_fr = var(pm, nw, :w)[f_bus]
    w_to = var(pm, nw, :w)[t_bus]

    tmsqr = [tm_fixed[i] ? tm[i]^2 : JuMP.@variable(pm.model, base_name="$(nw)_tmsqr_$(trans_id)_$(f_connections[i])", start=JuMP.start_value(tm[i])^2, lower_bound=JuMP.has_lower_bound(tm[i]) ? JuMP.lower_bound(tm[i])^2 : 0.9^2, upper_bound=JuMP.has_upper_bound(tm[i]) ? JuMP.upper_bound(tm[i])^2 : 1.1^2) for i in 1:length(tm)]

    for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
        if tm_fixed[idx]
            JuMP.@constraint(pm.model, w_fr[fc] == (pol*tm_scale*tm[idx])^2*w_to[tc])
        else
            PMD.PolyhedralRelaxations.construct_univariate_relaxation!(pm.model, x->x^2, tm[idx], tmsqr[idx], [JuMP.has_lower_bound(tm[idx]) ? JuMP.lower_bound(tm[idx]) : 0.9, JuMP.has_upper_bound(tm[idx]) ? JuMP.upper_bound(tm[idx]) : 1.1], false)

            tmsqr_w_to = JuMP.@variable(pm.model, base_name="$(nw)_tmsqr_w_to_$(trans_id)_$(t_bus)_$(tc)")
            PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(pm.model, tmsqr[idx], w_to[tc], tmsqr_w_to, [JuMP.lower_bound(tmsqr[idx]), JuMP.upper_bound(tmsqr[idx])], [JuMP.has_lower_bound(w_to[tc]) ? JuMP.lower_bound(w_to[tc]) : 0.0, JuMP.has_upper_bound(w_to[tc]) ? JuMP.upper_bound(w_to[tc]) : 1.1^2])

            JuMP.@constraint(pm.model, w_fr[fc] == (pol*tm_scale)^2*tmsqr_w_to)

            # with regcontrol
            if haskey(transformer,"controls")
                # TODO: fix LPUBFDiag version of transformer controls for on_off
                # v_ref = transformer["controls"]["vreg"][idx]
                # δ = transformer["controls"]["band"][idx]
                # r = transformer["controls"]["r"][idx]
                # x = transformer["controls"]["x"][idx]

                # # linearized voltage squared: w_drop = (2⋅r⋅p+2⋅x⋅q)
                # w_drop = JuMP.@expression(pm.model, 2*r*p_to[idx] + 2*x*q_to[idx])

                # # (v_ref-δ)^2 ≤ w_fr-w_drop ≤ (v_ref+δ)^2
                # # w_fr/1.1^2 ≤ w_to ≤ w_fr/0.9^2
                # w_drop_z_block = JuMP.@variable(pm.model, base_name="$(nw)_w_drop_z_block_$(trans_id)_$(idx)")

                # _IM.relaxation_product(pm.model, w_drop, z_block, w_drop_z_block; default_x_domain=(0.9^2, 1.1^2), default_y_domain=(0, 1))

                # w_fr_lb = JuMP.has_lower_bound(w_fr[fc]) && JuMP.lower_bound(w_fr[fc]) > 0 ? JuMP.lower_bound(w_fr[fc]) : 0.9^2
                # w_fr_ub = JuMP.has_upper_bound(w_fr[fc]) && isfinite(JuMP.upper_bound(w_fr[fc])) ? JuMP.upper_bound(w_fr[fc]) : 1.1^2

                # JuMP.@constraint(pm.model, w_fr[fc] ≥ z_block * (v_ref - δ)^2 + w_drop_z_block)
                # JuMP.@constraint(pm.model, w_fr[fc] ≤ z_block * (v_ref + δ)^2 + w_drop_z_block)
                # JuMP.@constraint(pm.model, w_fr[fc]/w_fr_ub ≤ w_to[tc])
                # JuMP.@constraint(pm.model, w_fr[fc]/w_fr_lb ≥ w_to[tc])
            end
        end
    end

    JuMP.@constraint(pm.model, p_fr + p_to .== 0)
    JuMP.@constraint(pm.model, q_fr + q_to .== 0)
end


@doc raw"""
    constraint_mc_transformer_power_yy_traditional_on_off(pm::PMD.LPUBFDiagModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)

    Links to and from power and voltages in a wye-wye transformer, assumes tm_fixed is true

```math
w_fr_i=(pol_i*tm_scale*tm_i)^2w_to_i
```
"""
function constraint_mc_transformer_power_yy_traditional_on_off(pm::PMD.LPUBFDiagModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    # z_voltage_fr = var(pm, nw, :z_voltage, f_bus)
    # z_voltage_to = var(pm, nw, :z_voltage, t_bus)

    tm = [tm_fixed[idx] ? tm_set[idx] : var(pm, nw, :tap, trans_id)[idx] for (idx,(fc,tc)) in enumerate(zip(f_connections,t_connections))]
    transformer = ref(pm, nw, :transformer, trans_id)

    p_fr = [var(pm, nw, :pt, f_idx)[p] for p in f_connections]
    p_to = [var(pm, nw, :pt, t_idx)[p] for p in t_connections]
    q_fr = [var(pm, nw, :qt, f_idx)[p] for p in f_connections]
    q_to = [var(pm, nw, :qt, t_idx)[p] for p in t_connections]

    w_fr = var(pm, nw, :w)[f_bus]
    w_to = var(pm, nw, :w)[t_bus]

    tmsqr = [tm_fixed[i] ? tm[i]^2 : JuMP.@variable(pm.model, base_name="$(nw)_tmsqr_$(trans_id)_$(f_connections[i])", start=JuMP.start_value(tm[i])^2, lower_bound=JuMP.has_lower_bound(tm[i]) ? JuMP.lower_bound(tm[i])^2 : 0.9^2, upper_bound=JuMP.has_upper_bound(tm[i]) ? JuMP.upper_bound(tm[i])^2 : 1.1^2) for i in 1:length(tm)]

    for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
        if tm_fixed[idx]
            JuMP.@constraint(pm.model, w_fr[fc] == (pol*tm_scale*tm[idx])^2*w_to[tc])
        else
            PMD.PolyhedralRelaxations.construct_univariate_relaxation!(pm.model, x->x^2, tm[idx], tmsqr[idx], [JuMP.has_lower_bound(tm[idx]) ? JuMP.lower_bound(tm[idx]) : 0.9, JuMP.has_upper_bound(tm[idx]) ? JuMP.upper_bound(tm[idx]) : 1.1], false)

            tmsqr_w_to = JuMP.@variable(pm.model, base_name="$(nw)_tmsqr_w_to_$(trans_id)_$(t_bus)_$(tc)")
            PMD.PolyhedralRelaxations.construct_bilinear_relaxation!(pm.model, tmsqr[idx], w_to[tc], tmsqr_w_to, [JuMP.lower_bound(tmsqr[idx]), JuMP.upper_bound(tmsqr[idx])], [JuMP.has_lower_bound(w_to[tc]) ? JuMP.lower_bound(w_to[tc]) : 0.0, JuMP.has_upper_bound(w_to[tc]) ? JuMP.upper_bound(w_to[tc]) : 1.1^2])

            JuMP.@constraint(pm.model, w_fr[fc] == (pol*tm_scale)^2*tmsqr_w_to)

            # with regcontrol
            if haskey(transformer,"controls")
                # TODO: fix LPUBFDiag version of transformer controls for on_off
                # v_ref = transformer["controls"]["vreg"][idx]
                # δ = transformer["controls"]["band"][idx]
                # r = transformer["controls"]["r"][idx]
                # x = transformer["controls"]["x"][idx]

                # # linearized voltage squared: w_drop = (2⋅r⋅p+2⋅x⋅q)
                # w_drop = JuMP.@expression(pm.model, 2*r*p_to[idx] + 2*x*q_to[idx])

                # # (v_ref-δ)^2 ≤ w_fr-w_drop ≤ (v_ref+δ)^2
                # # w_fr/1.1^2 ≤ w_to ≤ w_fr/0.9^2
                # w_drop_z_block = JuMP.@variable(pm.model, base_name="$(nw)_w_drop_z_block_$(trans_id)_$(idx)")

                # _IM.relaxation_product(pm.model, w_drop, z_block, w_drop_z_block; default_x_domain=(0.9^2, 1.1^2), default_y_domain=(0, 1))

                # w_fr_lb = JuMP.has_lower_bound(w_fr[fc]) && JuMP.lower_bound(w_fr[fc]) > 0 ? JuMP.lower_bound(w_fr[fc]) : 0.9^2
                # w_fr_ub = JuMP.has_upper_bound(w_fr[fc]) && isfinite(JuMP.upper_bound(w_fr[fc])) ? JuMP.upper_bound(w_fr[fc]) : 1.1^2

                # JuMP.@constraint(pm.model, w_fr[fc] ≥ z_block * (v_ref - δ)^2 + w_drop_z_block)
                # JuMP.@constraint(pm.model, w_fr[fc] ≤ z_block * (v_ref + δ)^2 + w_drop_z_block)
                # JuMP.@constraint(pm.model, w_fr[fc]/w_fr_ub ≤ w_to[tc])
                # JuMP.@constraint(pm.model, w_fr[fc]/w_fr_lb ≥ w_to[tc])
            end
        end
    end

    JuMP.@constraint(pm.model, p_fr + p_to .== 0)
    JuMP.@constraint(pm.model, q_fr + q_to .== 0)
end


"""
    constraint_mc_storage_losses_block_on_off(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, bus::Int, connections::Vector{Int}, r::Real, x::Real, p_loss::Real, q_loss::Real)

Neglects the active and reactive loss terms associated with the squared current magnitude.
"""
function constraint_mc_storage_losses_block_on_off(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, bus::Int, connections::Vector{Int}, ::Real, ::Real, p_loss::Real, q_loss::Real)
    z_block = var(pm, nw, :z_block, ref(pm, nw, :storage_block_map, i))

    ps = var(pm, nw, :ps, i)
    qs = var(pm, nw, :qs, i)
    sc = var(pm, nw, :sc, i)
    sd = var(pm, nw, :sd, i)
    qsc = var(pm, nw, :qsc, i)

    qsc_zblock = JuMP.@variable(pm.model, base_name="$(nw)_qd_zblock_$(i)")

    JuMP.@constraint(pm.model, qsc_zblock >= JuMP.lower_bound(qsc) * z_block)
    JuMP.@constraint(pm.model, qsc_zblock >= JuMP.upper_bound(qsc) * z_block + qsc - JuMP.upper_bound(qsc))
    JuMP.@constraint(pm.model, qsc_zblock <= JuMP.upper_bound(qsc) * z_block)
    JuMP.@constraint(pm.model, qsc_zblock <= qsc + JuMP.lower_bound(qsc) * z_block - JuMP.lower_bound(qsc))

    JuMP.@constraint(pm.model, sum(ps[c] for c in connections) + (sd - sc) == p_loss * z_block)
    JuMP.@constraint(pm.model, sum(qs[c] for c in connections) == qsc_zblock + q_loss * z_block)
end


"""
    constraint_mc_storage_losses_traditional_on_off(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, bus::Int, connections::Vector{Int}, r::Real, x::Real, p_loss::Real, q_loss::Real)

Neglects the active and reactive loss terms associated with the squared current magnitude.
"""
function constraint_mc_storage_losses_traditional_on_off(pm::PMD.LPUBFDiagModel, nw::Int, i::Int, bus::Int, connections::Vector{Int}, r::Real, x::Real, p_loss::Real, q_loss::Real)
    z_storage = var(pm, nw, :z_storage, i)

    ps = var(pm, nw, :ps, i)
    qs = var(pm, nw, :qs, i)
    sc = var(pm, nw, :sc, i)
    sd = var(pm, nw, :sd, i)
    qsc = var(pm, nw, :qsc, i)

    qsc_z_storage = JuMP.@variable(pm.model, base_name="$(nw)_qd_z_storage_$(i)")

    JuMP.@constraint(pm.model, qsc_z_storage >= JuMP.lower_bound(qsc) * z_storage)
    JuMP.@constraint(pm.model, qsc_z_storage >= JuMP.upper_bound(qsc) * z_storage + qsc - JuMP.upper_bound(qsc))
    JuMP.@constraint(pm.model, qsc_z_storage <= JuMP.upper_bound(qsc) * z_storage)
    JuMP.@constraint(pm.model, qsc_z_storage <= qsc + JuMP.lower_bound(qsc) * z_storage - JuMP.lower_bound(qsc))

    JuMP.@constraint(pm.model, sum(ps[c] for c in connections) + (sd - sc) == p_loss * z_storage)
    JuMP.@constraint(pm.model, sum(qs[c] for c in connections) == qsc_z_storage + q_loss * z_storage)
end


@doc raw"""
    constraint_storage_complementarity_mi_block_on_off(pm::PMD.LPUBFDiagModel, n::Int, i::Int, charge_ub::Real, discharge_ub::Real)

linear storage complementarity mi constraint for block mld problem

math```
sc_{on} + sd_{on} == z_{block}
```
"""
function constraint_storage_complementarity_mi_block_on_off(pm::PMD.LPUBFDiagModel, n::Int, i::Int, charge_ub::Real, discharge_ub::Real)
    sc = var(pm, n, :sc, i)
    sd = var(pm, n, :sd, i)
    sc_on = var(pm, n, :sc_on, i)
    sd_on = var(pm, n, :sd_on, i)

    z_block = var(pm, n, :z_block, ref(pm, n, :storage_block_map, i))

    JuMP.@constraint(pm.model, sc_on + sd_on == z_block)
    JuMP.@constraint(pm.model, sc_on*charge_ub >= sc)
    JuMP.@constraint(pm.model, sd_on*discharge_ub >= sd)
end


@doc raw"""
    constraint_storage_complementarity_mi_traditional_on_off(pm::PMD.LPUBFDiagModel, n::Int, i::Int, charge_ub::Real, discharge_ub::Real)

linear storage complementarity mi constraint for traditional mld problem

math```
sc_{on} + sd_{on} == z_{block}
```
"""
function constraint_storage_complementarity_mi_traditional_on_off(pm::PMD.LPUBFDiagModel, n::Int, i::Int, charge_ub::Real, discharge_ub::Real)
    sc = var(pm, n, :sc, i)
    sd = var(pm, n, :sd, i)
    sc_on = var(pm, n, :sc_on, i)
    sd_on = var(pm, n, :sd_on, i)

    z_storage = var(pm, n, :z_storage, i)

    JuMP.@constraint(pm.model, sc_on + sd_on == z_storage)
    JuMP.@constraint(pm.model, sc_on*charge_ub >= sc)
    JuMP.@constraint(pm.model, sd_on*discharge_ub >= sd)
end

