if {$argc != 2} {
    error "Usage: vivado -mode batch -source report_timing_from_dcp.tcl -tclargs DCP REPORT_DIR"
}

set dcp [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
open_checkpoint $dcp
create_clock -name ap_clk -period 10.000 [get_ports ap_clk]
report_timing_summary -delay_type max -max_paths 10 -report_unconstrained -file [file join $report_dir timing_summary.rpt]
write_checkpoint -force $dcp
