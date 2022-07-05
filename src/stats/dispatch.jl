"""
    get_timestep_voltage_statistics!(
        args::Dict{String,<:Any}
    )::Dict{String,Vector{Real}}

Gets voltage statistics min, mean, max for each timestep in-place in args, for use in [`entrypoint`][@ref entrypoint],
using [`get_timestep_voltage_statistics`](@ref get_timestep_voltage_statistics)
"""
function get_timestep_voltage_statistics!(args::Dict{String,<:Any})::Dict{String,Vector{Real}}
    args["output_data"]["Voltages"] = get_timestep_voltage_statistics(get(get(args, "optimal_dispatch_result", Dict{String,Any}()), "solution", Dict{String,Any}()), args["network"])
end


"""
    get_timestep_voltage_statistics(::Dict{String,<:Any}, ::String)

Helper function for the variant where `args["network"]` hasn't been parsed yet.
"""
get_timestep_voltage_statistics(::Dict{String,<:Any}, ::String) = Dict{String,Vector{Real}}(
    "Min voltage (p.u.)" => Real[],
    "Mean voltage (p.u.)" => Real[],
    "Max voltage (p.u.)" => Real[],
)


"""
    get_voltage_min_mean_max(
        solution::Dict{String,<:Any},
        data::Dict{String,<:Any};
        make_per_unit::Bool=true
    )::Tuple{Real,Real,Real}

Calculates the minimum, mean, and maximum of the voltages across a network (not a multinetwork)

`data` is used to convert the units to per_unit if `make_per_unit` and the data is not already per_unit.

If `make_per_unit` (default: true), will return voltage statistics in per-unit representation.
If `make_per_unit` is false, and there are different voltage bases across the network, the statistics will
not make sense.
"""
function get_voltage_min_mean_max(solution::Dict{String,<:Any}, data::Dict{String,<:Any}; make_per_unit::Bool=true)::Tuple{Real,Real,Real}
    if make_per_unit
        bus_vbase, line_vbase = PMD.calc_voltage_bases(data, data["settings"]["vbases_default"])
        voltages = [get(bus, "vm", zeros(length(data["bus"][id]["terminals"]))) ./ bus_vbase[id] for (id,bus) in get(solution, "bus", Dict())]
    else
        voltages = [get(bus, "vm", zeros(length(data["bus"][id]["terminals"]))) for (id,bus) in get(solution, "bus", Dict())]
    end

    return isempty(voltages) ? (NaN, NaN, NaN) : (minimum(minimum.(voltages)), Statistics.mean(Statistics.mean.(voltages)), maximum(maximum.(voltages)))
end


"""
    get_timestep_voltage_statistics(
        solution::Dict{String,<:Any},
        network::Dict{String,<:Any};
        make_per_unit::Bool=true
    )::Dict{String,Vector{Real}}

Returns statistics on the Minimum, Mean, and Maximum voltages for each timestep using
[`get_voltage_min_mean_max`](@ref get_voltage_min_mean_max)

If `make_per_unit` (default: true), will return voltage statistics in per-unit representation.
If `make_per_unit` is false, and there are different voltage bases across the network, the
statistics will not make sense.
"""
function get_timestep_voltage_statistics(solution::Dict{String,<:Any}, network::Dict{String,<:Any}; make_per_unit::Bool=true)::Dict{String,Vector{Real}}
    voltages = Dict{String,Vector{Real}}(
        "Min voltage (p.u.)" => Real[],
        "Mean voltage (p.u.)" => Real[],
        "Max voltage (p.u.)" => Real[],
    )
    for n in sort([parse(Int,i) for i in keys(get(solution, "nw", Dict()))])
        per_unit = get(solution["nw"]["$n"], "per_unit", true)

        nw = network["nw"]["$n"]
        nw["data_model"] = network["data_model"]
        min_v, mean_v, max_v = get_voltage_min_mean_max(solution["nw"]["$n"], nw; make_per_unit=make_per_unit && !per_unit)
        push!(voltages["Min voltage (p.u.)"], min_v)
        push!(voltages["Mean voltage (p.u.)"], mean_v)
        push!(voltages["Max voltage (p.u.)"], max_v)
    end

    return voltages
end


"""
    get_timestep_dispatch!(
        args::Dict{String,<:Any}
    )::Vector{Dict{String,Any}}

Gets the optimal dispatch results in-place in args, for use in [`entrypoint`](@ref entrypoint), using
[`get_timestep_dispatch`](@ref get_timestep_dispatch).
"""
function get_timestep_dispatch!(args::Dict{String,<:Any})::Vector{Dict{String,Any}}
    args["output_data"]["Powerflow output"] = get_timestep_dispatch(get(get(args, "optimal_dispatch_result", Dict{String,Any}()), "solution", Dict{String,Any}()), get(args, "network", Dict{String,Any}()))
end


"""
    get_timestep_dispatch(::Dict{String,<:Any}, ::String)::Vector{Dict{String,Any}}

Helper function for the variant where `args["network"]` hasn't been parsed yet.
"""
get_timestep_dispatch(::Dict{String,<:Any}, ::String)::Vector{Dict{String,Any}} = Dict{String,Any}[]


"""
    get_timestep_dispatch(
        solution::Dict{String,<:Any},
        network::Dict{String,<:Any}
    )::Vector{Dict{String,Any}}

Returns the dispatch information for generation assets (generator, storage, solar, voltage_source) and
bus voltage magnitudes in SI units for each timestep from the optimal dispatch `solution`
"""
function get_timestep_dispatch(solution::Dict{String,<:Any}, network::Dict{String,<:Any})::Vector{Dict{String,Any}}
    dispatch = Dict{String,Any}[]

    for n in sort([parse(Int, i) for i in keys(get(solution, "nw", Dict()))])
        _dispatch = Dict{String,Any}(
            "bus" => Dict{String,Any}(),
        )

        for (gen_type, (p, q)) in [("storage", ("ps", "qs")), ("generator", ("pg", "qg")), ("solar", ("pg", "qg")), ("voltage_source", ("pg", "qg"))]
            if !isempty(get(network["nw"]["$n"], gen_type, Dict()))
                _gen_type_sol = get(solution["nw"]["$n"], gen_type, Dict())
                _dispatch[gen_type] = Dict{String,Any}()
                for (id, _gen) in network["nw"]["$n"][gen_type]
                    conns = _gen["connections"]
                    gen = get(_gen_type_sol, id, Dict())
                    _dispatch[gen_type][id] = Dict{String,Any}(
                        "real power setpoint (kW)" => get(gen, p, zeros(length(conns))),
                        "reactive power setpoint (kVar)" => get(gen, q, zeros(length(conns))),
                        "connections" => conns,
                    )
                end
            end
        end

        for (id, _bus) in get(network["nw"]["$n"], "bus", Dict())
            bus = get(get(solution["nw"]["$n"], "bus", Dict()), id, Dict())
            terms = _bus["terminals"]
            _dispatch["bus"][id] = Dict{String,Any}(
                "voltage (V)" => haskey(bus, "vr") && haskey(bus, "vi") ? sqrt.(bus["vr"].^2 + bus["vi"].^2) : haskey(bus, "w") ? sqrt.(bus["w"]) : get(bus, "vm", zeros(length(terms))),
                "terminals" => terms,
            )
        end

        if !isempty(get(network["nw"]["$n"], "switch", Dict()))
            _dispatch["switch"] = Dict{String,Any}()
        end

        for (id, _switch) in get(network["nw"]["$n"], "switch", Dict())
            connections = _switch["f_connections"]

            switch = get(get(solution["nw"]["$n"], "switch", Dict()), id, Dict())

            f_bus_id = _switch["f_bus"]
            terminals = network["nw"]["$n"]["bus"][f_bus_id]["terminals"]

            bus = get(get(solution["nw"]["$n"], "bus", Dict()), f_bus_id, Dict())

            _dispatch["switch"][id] = Dict{String,Any}(
                "real power flow (kW)" => get(switch, "pf", zeros(length(connections))),
                "reactive power flow (kVar)" => get(switch, "qf", zeros(length(connections))),
                "voltage (V)" => haskey(bus, "vr") && haskey(bus, "vi") ? sqrt.(bus["vr"].^2 + bus["vi"].^2)[[findfirst(isequal(c), terminals) for c in connections]] : haskey(bus, "w") ? sqrt.(bus["w"])[[findfirst(isequal(c), terminals) for c in connections]] : haskey(bus, "vm") ? bus["vm"][[findfirst(isequal(c), terminals) for c in connections]] : zeros(length(terminals)),  # TODO: wrong units, actually kV
                "connections" => connections,
            )
            current = _dispatch["switch"][id]["real power flow (kW)"] ./ _dispatch["switch"][id]["voltage (V)"]
            current[isnan.(current)] .= 0.0
            _dispatch["switch"][id]["current (A)"] = current
        end

        push!(dispatch, _dispatch)
    end

    return dispatch
end


"""
    get_timestep_inverter_states!(args::Dict{String,<:Any})::Vector{Dict{String,Any}}

Adds field "inverter" to power flow output for inverter objects, i.e., storage
generator, voltage_source, solar. See [`get_timestep_inverter_states`](@ref get_timestep_inverter_states)
"""
function get_timestep_inverter_states!(args::Dict{String,<:Any})::Vector{Dict{String,Any}}
    states = get_timestep_inverter_states(get(args, "optimal_switching_results", Dict{String,Any}()))

    _powerflow_outputs = Dict{String,Any}[]
    powerflow_outputs = get(args["output_data"], "Powerflow output", Dict{String,Any}())
    if !isempty(powerflow_outputs) && !isempty(states) && length(powerflow_outputs) == length(states)
        for (pfo, state) in zip(powerflow_outputs, states)
            push!(_powerflow_outputs, recursive_merge_including_vectors(pfo, state))
        end
    elseif !isempty(powerflow_outputs) && !isempty(states) && length(powerflow_outputs) != length(states)
        @error "size of inverter states timeline and powerflow timeline are different"
    else
        _powerflow_outputs = states
    end

    args["output_data"]["Powerflow output"] = _powerflow_outputs

    return states
end


"""
    get_timestep_inverter_states(optimal_switching_results::Dict{String,<:Any})::Vector{Dict{String,Any}}

Gets 'inverter' state for each generation object at each timestep from `optimal_switching_results`. Defaults
to `GRID_FORMING` if no inverter state is available.
"""
function get_timestep_inverter_states(optimal_switching_results::Dict{String,<:Any})::Vector{Dict{String,Any}}
    timesteps = Dict{String,Any}[]
    for n in sort(parse.(Int, collect(keys(optimal_switching_results))))
        sol = get(optimal_switching_results["$n"], "solution", Dict())
        timestep = Dict{String,Any}()
        for t in ["generator", "solar", "storage", "voltage_source"]
            if haskey(sol, t) && !isempty(sol[t])
                timestep[t] = Dict{String,Any}()
                for (i,obj) in sol[t]
                    timestep[t][i] = Dict{String,Any}("inverter"=>string(get(obj, "inverter", GRID_FORMING)))
                end
            end
        end

        push!(timesteps, timestep)
    end

    return timesteps
end


"""
    get_timestep_dispatch_optimization_metadata!(
        args::Dict{String,<:Any}
    )::Dict{String,Any}

Retrieves the switching optimization results metadata from the optimal switching solution via
[`get_timestep_dispatch_optimization_metadata`](@ref get_timestep_dispatch_optimization_metadata)
and applies it in-place to args, for use with [`entrypoint`](@ref entrypoint)
"""
function get_timestep_dispatch_optimization_metadata!(args::Dict{String,<:Any})::Dict{String,Any}
    args["output_data"]["Optimal dispatch metadata"] = get_timestep_dispatch_optimization_metadata(get(args, "optimal_dispatch_result", Dict{String,Any}()))
end


"""
    get_timestep_dispatch_optimization_metadata(
        optimal_dispatch_result::Dict{String,Any}
    )::Dict{String,Any}

Gets the metadata from the optimal switching results for each timestep, returning a list of Dicts
(if `opt_switch_algorithm="rolling-horizon"`), or a list with a single Dict (if `opt_switch_algorithm="full-lookahead"`).
"""
function get_timestep_dispatch_optimization_metadata(optimal_dispatch_result::Dict{String,Any})::Dict{String,Any}
    return _sanitize_results_metadata!(filter(x->x.first!="solution", optimal_dispatch_result))
end
