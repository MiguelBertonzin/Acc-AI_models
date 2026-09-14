# Exporta uma solução Vitis HLS já sintetizada como IP Catalog.
# Execute dentro da raiz do build, sem espaços no caminho:
#   vitis_hls -f export_existing_hls_ip.tcl

set project_name "resnet8_resource_fifo_opt"
set project_dir "${project_name}_prj"
set solution_name "solution1"
set ip_version "1.0.0"

if {![file isdirectory $project_dir]} {
    error "Projeto HLS não encontrado: [file normalize $project_dir]"
}

if {![file isdirectory "$project_dir/$solution_name/syn/verilog"]} {
    error "RTL sintetizado não encontrado em $project_dir/$solution_name/syn/verilog"
}

open_project $project_dir
open_solution $solution_name

puts "EXPORT_IP_PROJECT=[file normalize $project_dir]"
puts "EXPORT_IP_SOLUTION=$solution_name"
puts "EXPORT_IP_VERSION=$ip_version"

export_design -format ip_catalog -version $ip_version

puts "EXPORT_IP_STATUS=PASS"
exit
