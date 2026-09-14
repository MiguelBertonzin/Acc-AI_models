# Valida um repositório IP, instancia o núcleo e gera seus targets.
# Argumentos: <repo> <project_dir> <expected_vlnv> <module_name>

if {$argc != 4} {
    error "Uso: vivado -source validate_ip_catalog.tcl -tclargs <repo> <project_dir> <expected_vlnv> <module_name>"
}

set repo [file normalize [lindex $argv 0]]
set check_dir [file normalize [lindex $argv 1]]
set expected_vlnv [lindex $argv 2]
set module_name [lindex $argv 3]

if {![file exists "$repo/component.xml"]} {
    error "component.xml não encontrado em $repo"
}

create_project -force ip_catalog_validation $check_dir \
    -part xczu7ev-ffvc1156-2-e

set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property IP_REPO_PATHS [list $repo] [current_project]
update_ip_catalog -rebuild

puts "IP_CATALOG_MATCHES:"
foreach definition [get_ipdefs -all -quiet *:hls:resnet8_resource_fifo_opt:*] {
    puts "  $definition"
}

if {[llength [get_ipdefs -all -quiet $expected_vlnv]] == 0} {
    error "VLNV não encontrado no catálogo: $expected_vlnv"
}

create_ip -vlnv $expected_vlnv -module_name $module_name
set ip [get_ips $module_name]

if {![validate_ip -save_ip $ip]} {
    error "validate_ip retornou falha para $expected_vlnv"
}

generate_target all $ip

puts "IP_CATALOG_STATUS=PASS"
puts "IP_CATALOG_VLNV=[get_property IPDEF $ip]"
puts "IP_CATALOG_XCI=[get_property IP_FILE $ip]"

close_project
exit
