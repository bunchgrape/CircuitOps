import openroad as ord
from openroad import Tech, Design, Timing
from OpenROAD_helper import *
import argparse
import rcx
import helpers
import os
from pathlib import Path

parser = argparse.ArgumentParser(description = "parsing the name of the benchmark")
parser.add_argument("--design_name", type = str, default = "NV_NVDLA_partition_m")
pyargs = parser.parse_args()
tech, design = load_design(pyargs.design_name, False)
rcx_rule = Path("../../platforms/ASAP7/rcx_patterns.rules")

signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
signal_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
clk_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
grt = design.getGlobalRouter()
grt.clear()
grt.setAllowCongestion(True)
grt.setMinRoutingLayer(signal_low_layer)
grt.setMaxRoutingLayer(signal_high_layer)
grt.setMinLayerForClock(clk_low_layer)
grt.setMaxLayerForClock(clk_high_layer)
grt.setAdjustment(0.5)
grt.setVerbose(False)
grt.globalRoute(False)
design.evalTclString("estimate_parasitics -global_routing")

# bench_spef_file = Path("../../designs/%s/%s.spef"%(pyargs.design_name, pyargs.design_name))
# cmd = "bench_read_spef %s"%bench_spef_file
# design.evalTclString(cmd)
# rcx.bench_wires(False, False, False, False, 100, 5, 100, -1, "1", "1 2 2.5 3 3.5 4 4.5 5 6 8 10 12", 100, 100)
# rcx.write_rules("asap7.rules", 'result', "TYP", 0)

rcx.define_process_corner(0, "X")
rcx.extract(str(rcx_rule), 1, 50, 0.1, 10, 5, "", False, False)

filename = '6_final.spef'
result_dir = os.path.join(os.getcwd(), 'result')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
root_ext = os.path.splitext(filename)
filename = "{}-py{}".format(*root_ext)
spef_file = os.path.join(result_dir, filename)

rcx.write_spef(spef_file, "", 0, False)