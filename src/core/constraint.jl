"max actions per timestep switch constraint"
function constraint_switch_state_max_actions(pm::PMD._PM.AbstractPowerModel, nw::Int)
    max_switch_changes = get(pm.data, "max_switch_changes", length(get(pm.data, "switch", Dict())))

    state_start = Dict(
        l => PMD.ref(pm, nw, :switch, l, "state") for l in PMD.ids(pm, nw, :switch)
    )

    PMD.JuMP.@constraint(pm.model, sum((state_start[l] - PMD.var(pm, nw, :switch_state, l)) * (round(state_start[l]) == 0 ? -1 : 1) for l in PMD.ids(pm, nw, :switch_dispatchable)) <= max_switch_changes)
end


""
function constraint_load_block_isolation(pm::PMD._PM.AbstractPowerModel, nw::Int; relax::Bool=true)
    for (b, block) in PMD.ref(pm, nw, :load_blocks)
        z_demand = PMD.var(pm, nw, :z_demand_blocks, b)
        for s in PMD.ref(pm, nw, :load_block_switches, b)
            z_switch = PMD.var(pm, nw, :switch_state, s)
            if relax
                PMD.JuMP.@constraint(pm.model, z_switch <= z_demand)
                PMD.JuMP.@constraint(pm.model, z_switch >= 0)
            else
                PMD.JuMP.@constraint(pm.model, !z_demand => {z_switch == 0})
            end
        end
    end
end
