source project.tcl
open_project ${project_name}_prj
open_solution solution1
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x -DRTL_SIM"
cosim_design -rtl verilog -trace_level none
set cfile [open tb_data/csim_results.log r]
set cresults [read $cfile]
close $cfile
set rfile [open tb_data/rtl_cosim_results.log r]
set rresults [read $rfile]
close $rfile
if {$cresults ne $rresults} {error "C/RTL output mismatch"}
export_design -format ip_catalog -version $version
exit
