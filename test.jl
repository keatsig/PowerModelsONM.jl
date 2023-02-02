import PowerModelsONM as ONM
import PowerModelsDistribution as PMD
import InfrastructureModels as IM
import JuMP
import HiGHS
import Cbc
import LinearAlgebra
import StatsBase as SB

PMD.silence!()
onm_path = joinpath(dirname(pathof(ONM)), "..")
eng = PMD.parse_file(joinpath(onm_path, "test/data/ieee13_feeder.dss"))
# eng["settings"]["sbase_default"] = 100
eng["switch_close_actions_ub"] = Inf
PMD.apply_voltage_bounds!(eng)
math = ONM.transform_data_model(eng)
pm = ONM.instantiate_onm_model(eng, PMD.LPUBFDiagPowerModel, ONM.build_block_mld).model;


all_scen_var = Dict(scen=> Dict() for scen=1:N_scenarios)
all_scen_var[scen]["z_block"] = JuMP.@variable(
                        model,
                        [i in keys(ref[:blocks])],
                        base_name="0_z_block_$(scen)",
                        lower_bound=0,
                        upper_bound=1,
                        binary=true
                    )

all_scen_var[scen]["w"] = Dict(
    i => JuMP.@variable(
        model,
        [t in bus["terminals"]],
        base_name="0_w_$(i)",
        lower_bound=0,
    ) for (i,bus) in ref[:bus]
)
