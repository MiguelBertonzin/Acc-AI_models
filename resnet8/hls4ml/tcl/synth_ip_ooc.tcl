# Executa síntese Out-of-Context do IP já instanciado em um projeto Vivado.
# Argumentos: <project.xpr> <module_name> <report_path>

if {$argc != 3} {
    error "Uso: vivado -source synth_ip_ooc.tcl -tclargs <project.xpr> <module_name> <report_path>"
}

set project_file [file normalize [lindex $argv 0]]
set module_name [lindex $argv 1]
set report_path [file normalize [lindex $argv 2]]
set run_name "${module_name}_synth_1"

open_project $project_file

if {[llength [get_ips -quiet $module_name]] == 0} {
    error "IP não encontrado no projeto: $module_name"
}

if {[llength [get_runs -quiet $run_name]] == 0} {
    create_ip_run [get_ips $module_name]
}

launch_runs $run_name -jobs 7
wait_on_run $run_name

set run_status [get_property STATUS [get_runs $run_name]]
set run_progress [get_property PROGRESS [get_runs $run_name]]
puts "IP_OOC_RUN=$run_name"
puts "IP_OOC_STATUS=$run_status"
puts "IP_OOC_PROGRESS=$run_progress"

if {![string match "*Complete*" $run_status] || $run_progress ne "100%"} {
    error "Síntese OOC não concluiu corretamente: $run_status ($run_progress)"
}

open_run $run_name
report_utilization -file $report_path
puts "IP_OOC_REPORT=$report_path"
puts "IP_OOC_VALIDATION=PASS"

close_project
exit
