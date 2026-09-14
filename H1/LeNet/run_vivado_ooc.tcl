if {$argc != 2} {
    error "Usage: vivado -mode batch -source run_vivado_ooc.tcl -tclargs BUILD_DIR RF_LABEL"
}

set build_dir [file normalize [lindex $argv 0]]
set rf_label [lindex $argv 1]
set project_name "lenet_mnist_cap64_hls"
set part "xczu7ev-ffvc1156-2-e"
set rtl_dir [file join $build_dir "${project_name}_prj" solution1 syn verilog]
set report_dir [file join $build_dir vivado_ooc]

if {![file isdirectory $rtl_dir]} {
    error "RTL directory not found: $rtl_dir"
}

file mkdir $report_dir
read_verilog [glob -nocomplain [file join $rtl_dir *.v]]
synth_design -top $project_name -part $part -mode out_of_context
create_clock -name ap_clk -period 10.000 [get_ports ap_clk]
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt

report_utilization -hierarchical -file [file join $report_dir utilization_hierarchical.rpt]
report_utilization -file [file join $report_dir utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -report_unconstrained -file [file join $report_dir timing_summary.rpt]
report_methodology -file [file join $report_dir methodology.rpt]
write_checkpoint -force [file join $report_dir "${project_name}_${rf_label}_post_synth.dcp"]

set out [open [file join $report_dir status.txt] w]
puts $out "status=completed"
puts $out "rf_label=$rf_label"
puts $out "part=$part"
puts $out "top=$project_name"
puts $out "mode=out_of_context_post_synthesis"
close $out
