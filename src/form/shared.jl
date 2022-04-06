@doc raw"""
    constraint_mc_bus_voltage_magnitude_sqr_block_on_off(
        pm::PMD.AbstractUnbalancedWModels,
        nw::Int,
        i::Int,
        vmin::Vector{<:Real},
        vmax::Vector{<:Real}
    )

on/off block bus voltage magnitude squared constraint for W models

```math
```
"""
function constraint_mc_bus_voltage_magnitude_sqr_block_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real})
    w = var(pm, nw, :w, i)
    z_block = var(pm, nw, :z_block, ref(pm, nw, :bus_block_map, i))

    terminals = ref(pm, nw, :bus, i)["terminals"]
    grounded = ref(pm, nw, :bus, i)["grounded"]

    for (idx,t) in [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
        isfinite(vmax[idx]) && JuMP.@constraint(pm.model, w[t] <= vmax[idx]^2*z_block)
        isfinite(vmin[idx]) && JuMP.@constraint(pm.model, w[t] >= vmin[idx]^2*z_block)
    end
end


@doc raw"""
    constraint_mc_bus_voltage_magnitude_sqr_traditional_on_off(
        pm::PMD.AbstractUnbalancedWModels,
        nw::Int,
        i::Int,
        vmin::Vector{<:Real},
        vmax::Vector{<:Real}
    )

on/off traditional bus voltage magnitude squared constraint for W models

```math
```
"""
function constraint_mc_bus_voltage_magnitude_sqr_traditional_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real})
    w = var(pm, nw, :w, i)
    z_voltage = var(pm, nw, :z_voltage, i)

    terminals = ref(pm, nw, :bus, i)["terminals"]
    grounded = ref(pm, nw, :bus, i)["grounded"]

    for (idx,t) in [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
        isfinite(vmax[idx]) && JuMP.@constraint(pm.model, w[t] <= vmax[idx]^2*z_voltage)
        isfinite(vmin[idx]) && JuMP.@constraint(pm.model, w[t] >= vmin[idx]^2*z_voltage)
    end
end


"""
    constraint_mc_bus_voltage_block_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real})

Redirects to `constraint_mc_bus_voltage_magnitude_sqr_block_on_off` for `AbstractUnbalancedWModels`
"""
constraint_mc_bus_voltage_block_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real}) = constraint_mc_bus_voltage_magnitude_sqr_block_on_off(pm, nw, i, vmin, vmax)


"""
    constraint_mc_bus_voltage_traditional_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real})

Redirects to `constraint_mc_bus_voltage_magnitude_sqr_traditional_on_off` for `AbstractUnbalancedWModels`
"""
constraint_mc_bus_voltage_traditional_on_off(pm::PMD.AbstractUnbalancedWModels, nw::Int, i::Int, vmin::Vector{<:Real}, vmax::Vector{<:Real}) = constraint_mc_bus_voltage_magnitude_sqr_traditional_on_off(pm, nw, i, vmin, vmax)


@doc raw"""
    constraint_mc_inverter_theta_ref(pm::PMD.AbstractUnbalancedPolarModels, nw::Int, i::Int, va_ref::Vector{<:Real})

Phase angle constraints at reference buses for the Unbalanced Polar models

math```
\begin{align*}
V_a - V^{ref}_a \leq 60^{\circ} * (1-\sum{z_{inv}})
V_a - V^{ref}_a \geq -60^{\circ} * (1-\sum{z_{inv}})
\end{align*}
```
"""
function constraint_mc_inverter_theta_ref(pm::PMD.AbstractUnbalancedPolarModels, nw::Int, i::Int, va_ref::Vector{<:Real})
    terminals = ref(pm, nw, :bus, i)["terminals"]
    va = var(pm, nw, :va, i)
    inverter_objects = ref(pm, nw, :bus_inverters, i)
    z_inverters = [var(pm, nw, :z_inverter, inv_obj) for inv_obj in inverter_objects]

    if !isempty(inverter_objects)
        for (idx,t) in enumerate(terminals)
            JuMP.@constraint(pm.model, va[t] - va_ref[idx] <=  deg2rad(60) * (1-sum(z_inverters)))
            JuMP.@constraint(pm.model, va[t] - va_ref[idx] >= -deg2rad(60) * (1-sum(z_inverters)))
        end
    end
end
