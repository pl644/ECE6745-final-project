#=========================================================================
# 04-cadence-innovus-pnr/run.tcl
#=========================================================================

#-------------------------------------------------------------------------
# Initial Setup
#-------------------------------------------------------------------------

set init_mmmc_file "setup-timing.tcl"
set init_verilog   "../02-synopsys-dc-synth/post-synth.v"
set init_top_cell  "{{design_name}}"
set init_lef_file  "$env(ECE6745_STDCELLS)/rtk-tech.lef $env(ECE6745_STDCELLS)/stdcells.lef"
set init_gnd_net   "VSS"
set init_pwr_net   "VDD"

init_design

# Set the process node to 45nm for technology-specific optimizations

setDesignMode -process 45

# Turn off signal integrity analysis (i.e., signal cross-talk)

setDelayCalMode -SIAware false

# Turn off useful clock skew (i.e., the tools will try to reduce the
# clock skew across all flip-flops)

setOptMode -usefulSkew false

# Add in an addition hold target slack to force Cadence Innovus to
# work harder; this way if it misses by a little we will still end up
# with positive hold slack.

setOptMode -holdTargetSlack 0.010

# Specify which standard cells to use when fixing hold time violations

setOptMode -holdFixingCells {
  BUF_X1 BUF_X1 BUF_X2 BUF_X4 BUF_X8 BUF_X16 BUF_X32
}

# Increase the number of significant digits in reports

set report_precision 4

#-------------------------------------------------------------------------
# Floorplanning
#-------------------------------------------------------------------------

# Assume an aspect ratio of 1 and a target placement density of 40%.
# Leave 4um border around the outside of the core area for the pad
# ring.

floorPlan -r 1.0 0.70 4.0 4.0 4.0 4.0

#-------------------------------------------------------------------------
# Power Routing
#-------------------------------------------------------------------------

globalNetConnect VDD -type pgpin -pin VDD -inst * -verbose
globalNetConnect VSS -type pgpin -pin VSS -inst * -verbose

# Route the M1 tracks for each row of standard cells

sroute -nets {VDD VSS}

# Route a power ring on M8 and M9 around the outside of the core area

addRing \
  -nets {VDD VSS} -width 0.8 -spacing 0.8 \
  -layer [list top 9 bottom 9 left 8 right 8]

# Route horizontal stripes on M9 for the power grid

addStripe \
  -nets {VSS VDD} -layer 9 -direction horizontal \
  -width 0.8 -spacing 4.8 \
  -set_to_set_distance 11.2 -start_offset 2.4

# Route vertical stripes on M for the power grid

addStripe \
  -nets {VSS VDD} -layer 8 -direction vertical \
  -width 0.8 -spacing 4.8 \
  -set_to_set_distance 11.2 -start_offset 2.4

#-------------------------------------------------------------------------
# Placement
#-------------------------------------------------------------------------

# Place the design with preliminary routing

place_design

# Try to place the input/output pins so they are close to the
# standard-cells they are connected to

assignIoPins -pin *

#-------------------------------------------------------------------------
# Clock-Tree Synthesis
#-------------------------------------------------------------------------

# Create the clock tree specification from the timing constraints

create_ccopt_clock_tree_spec

# Turn off an optimization that allows Cadence Innovus to decide to
# add non-zero clock source insertion latency. Turning this on will
# complicate back-annotated gate-level simulation.

set_ccopt_property update_io_latency false

# Synthesize the clock tree

ccopt_design -cts

# Fix setup time violations by reducing the delay of the slow paths

optDesign -postCTS -setup

# Fix hold time violations by increasing the delay of the fast paths

optDesign -postCTS -hold

#-------------------------------------------------------------------------
# Routing
#-------------------------------------------------------------------------

# Route the design

routeDesign

# Fix setup time violations by reducing the delay of the slow paths

optDesign -postRoute -setup

# Fix hold time violations by increasing the delay of the fast paths

optDesign -postRoute -hold

# Fix design rule violations

optDesign -postRoute -drv

# Extract the interconnect parasitics

extractRC

#-------------------------------------------------------------------------
# Finishing
#-------------------------------------------------------------------------

# Add filler cells to complete each row of standard cells

setFillerMode -core {FILLCELL_X4 FILLCELL_X2 FILLCELL_X1}
addFiller

# Check the final physical design

verifyConnectivity
verify_drc

#-------------------------------------------------------------------------
# Outputs
#-------------------------------------------------------------------------

# Output the design which can be loaded into Cadence Innovus later

saveDesign post-pnr.enc

# Output the post-pnr gate-level netlist

saveNetlist post-pnr.v

# Output the interconnect parasitics in SPEF format for power analsys

rcOut -rc_corner typical -spef post-pnr.spef

# Output the timing delays in SDF format for back-annotated gate-level sim

write_sdf post-pnr.sdf

# Output the SDC file so we can check for clock source insertion latency

write_sdc post-pnr.sdc

# Merge the standard-cell layout with the place and routed design to
# stream out the final layout for the block

streamOut post-pnr.gds \
  -merge "$env(ECE6745_STDCELLS)/stdcells.gds" \
  -mapFile "$env(ECE6745_STDCELLS)/rtk-stream-out.map"

# Report the critical path (i.e., setup-time constraint)

report_timing -late  -path_type full_clock -net > timing-setup.rpt

# Report the short path (i.e., hold-time constraint)

report_timing -early -path_type full_clock -net > timing-hold.rpt

# Report the area broken down by module

report_area > area.rpt

exit
