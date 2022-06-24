
"string to PowerModelsDistribution type conversion for opt-disp-formulation"
const _formulation_lookup = Dict{String,Type}(
    "acr" => PMD.ACRUPowerModel,
    "acrupowermodel" => PMD.ACRUPowerModel,
    "acp" => PMD.ACPUPowerModel,
    "acpupowermodel" => PMD.ACPUPowerModel,
    "lindistflow" => PMD.LPUBFDiagPowerModel,
    "lpubfdiag" => PMD.LPUBFDiagPowerModel,
    "lpubf" => PMD.LPUBFDiagPowerModel,
    "lpubfdiagpowermodel" => PMD.LPUBFDiagPowerModel,
    "nfa" => PMD.NFAUPowerModel,
    "nfau" => PMD.NFAUPowerModel,
    "nfaupowermodel" => PMD.NFAUPowerModel,
    "fot" => PMD.FOTRUPowerModel,
    "fotr" => PMD.FOTRUPowerModel,
    "fotp" => PMD.FOTPUPowerModel,
    "fbs" => PMD.FBSUBFPowerModel,
)


"""
    _get_formulation(form_string::String)

helper function to convert from opt-disp-formulation, opt-switch-formulation string to PowerModelsDistribution Type
"""
_get_formulation(form_string::String)::Type = _formulation_lookup[lowercase(form_string)]


"""
    _get_formulation(form::Type)

helper function to convert from PowerModelsDistribution Type to PowerModelsDistribution Type
"""
_get_formulation(form::Type)::Type = form


"Inverter Enum to indicate device is operating in grid-follow or grid-forming mode"
@enum Inverter GRID_FOLLOWING GRID_FORMING
@doc "Inverter acting as grid-following" GRID_FOLLOWING
@doc "Inverter acting as grid-forming" GRID_FORMING


"""
    Base.parse(::Type{T}, inverter::String)::T where T <: Inverter

Parses the 'inverter' property from dss settings schema into an Inverter enum
"""
function Base.parse(::Type{T}, inverter::String)::T where T <: Inverter
    if lowercase(inverter) ∈ ["grid_forming", "gfm"]
        return GRID_FORMING
    elseif lowercase(inverter) ∈ ["grid_following", "gfl"]
        return GRID_FOLLOWING
    end

    @warn "inverter code '$inverter' not recognized, defaulting to GRID_FORMING"
    return GRID_FORMING
end

"Parses different options for Inverter enums in the settings schema"
Base.parse(::Type{Inverter}, inverter::Inverter)::Inverter = inverter
Base.parse(::Type{Inverter}, status::Bool)::Inverter = Inverter(Int(status))
Base.parse(::Type{Inverter}, status::Int)::Inverter = Inverter(status)
