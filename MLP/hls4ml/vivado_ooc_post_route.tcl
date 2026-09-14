set script_dir [file dirname [file normalize [info script]]]
set generated_dir [file join $script_dir mlp_iris_apfixed16_6_rf1_100mhz]
set rtl_dir [file join $generated_dir mlp_iris_prj solution1 syn verilog]
set report_dir [file join $generated_dir reports vivado]
file mkdir $report_dir

set part xczu7ev-ffvc1156-2-e
set top mlp_iris
set clock_period_ns 10.0

add_files [glob -directory $rtl_dir *.v]
synth_design -top $top -part $part -mode out_of_context
create_clock -name ap_clk -period $clock_period_ns [get_ports ap_clk]
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt
report_utilization -file [file join $report_dir utilization_post_synth_ooc.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $report_dir timing_post_synth_ooc.rpt]
write_checkpoint -force [file join $report_dir mlp_iris_post_synth_ooc.dcp]

place_design
phys_opt_design
route_design
report_utilization -file [file join $report_dir utilization_post_route.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -report_unconstrained -file [file join $report_dir timing_post_route.rpt]
report_drc -file [file join $report_dir drc_post_route.rpt]
write_checkpoint -force [file join $report_dir mlp_iris_post_route.dcp]

set timing_paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $timing_paths] > 0} {
    set worst_path [lindex $timing_paths 0]
    set wns [get_property SLACK $worst_path]
    set data_path_delay [get_property DATAPATH_DELAY $worst_path]
    set requirement [get_property REQUIREMENT $worst_path]
    set summary_file [open [file join $report_dir run_summary.txt] w]
    puts $summary_file "part=$part"
    puts $summary_file "top=$top"
    puts $summary_file "clock_period_ns=$clock_period_ns"
    puts $summary_file "wns_ns=$wns"
    puts $summary_file "data_path_delay_ns=$data_path_delay"
    puts $summary_file "requirement_ns=$requirement"
    close $summary_file
}
