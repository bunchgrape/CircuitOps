# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


proc check_is_in_clk {inst clk_nets} {
  set is_in_clk 0
  set cell_nets [$inst getITerms]
  foreach cell_net $cell_nets {
    set pin_nets [$cell_net getNet]
    foreach pin_net $pin_nets {
      foreach clk_net $clk_nets {
        if {$pin_net == $clk_net} {
          set is_in_clk 1
          break
        }
      }
      if {$is_in_clk eq 1} {
        break
      }
    }
    if {$is_in_clk eq 1} {
      break
    }
  }
  return $is_in_clk
}

proc get_net_route_length {net} {
  set net_route_length 0
  set net_name [$net getName]

  if {[::sta::Net_is_power [get_net $net_name]] || [::sta::Net_is_ground [get_net $net_name]]} {
    set swire [$net getSWires]
    set wires [$swire getWires]
  } else {
    set wires [$net getWire]
  }
  foreach wire $wires {
    if {$wire eq "NULL"} {
      continue
    } else {
      set wire_length [$wire getLength]
      set net_route_length [expr {$net_route_length + $wire_length}]
    }
  }
  return $net_route_length
}

proc get_pin_arr {pin_arg rf} {
  set pin [::sta::get_port_pin_error "pin" $pin_arg]
  set pin_arr {}
  foreach vertex [$pin vertices] {
    if { $vertex != "NULL" } {
      if {$rf == "rise"} {
        set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" "NULL" "rise" "arrive"]
      } else {
        set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" "NULL" "rise" "hold"]
      }
      if {$tmp_pin_arr != ""} {
        lappend pin_arr $tmp_pin_arr
      }

      if {$rf == "rise"} {
        set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" [::sta::default_arrival_clock] "rise" "arrive"]
      } else {
        set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" [::sta::default_arrival_clock] "rise" "hold"]
      }
      if {$tmp_pin_arr != ""} {
        lappend pin_arr $tmp_pin_arr
      }
      foreach clk [all_clocks] {
        if {$rf == "rise"} {
          set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" $clk "rise" "arrive"]
        } else {
          set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" $clk "rise" "hold"]
        }
        if {$tmp_pin_arr != ""} {
          lappend pin_arr $tmp_pin_arr
        }

        if {$rf == "rise"} {
          set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" $clk "fall" "arrive"]
        } else {
          set tmp_pin_arr [get_pin_arr_time $vertex "arrivals_clk_delays" $clk "fall" "hold"]
        }
        if {$tmp_pin_arr != ""} {
          lappend pin_arr $tmp_pin_arr
        }
      }
    }
  }
  return $pin_arr
}

proc get_pin_arr_time {vertex what clk clk_rf arrive_hold} {
  global sta_report_default_digits
  set rise [$vertex $what rise $clk $clk_rf $sta_report_default_digits]
  set fall [$vertex $what fall $clk $clk_rf $sta_report_default_digits]
  # Filter INF/-INF arrivals.
  if { !([::sta::delays_are_inf $rise] && [::sta::delays_are_inf $fall]) } {
    if {$clk != "NULL"} {
      set clk_str " ([get_name $clk] [::sta::rf_short_name $clk_rf])"
    } else {
      set clk_str ""
    }
    if {$arrive_hold == "arrive"} {
      set rise_fmt [::sta::format_delays $rise]
      return "$clk_str r $rise_fmt"
    } else {
      set fall_fmt [::sta::format_delays $fall]
      return "$clk_str f $fall_fmt"
    }
  }   
}


proc get_pin_slew {pin_arg} {
  global sta_report_default_digits
  set corner [::sta::parse_corner_or_all keys]
  set pin [::sta::get_port_pin_error "pin" $pin_arg]
  set digits $sta_report_default_digits
  set pin_slew {}
  foreach vertex [$pin vertices] {
    if { $corner == "NULL" } {
      set pin_slew_ "[::sta::rise_short_name] [::sta::format_time [$vertex slew rise min] $digits]:[::sta::format_time [$vertex slew rise max] $digits] [::sta::fall_short_name] [::sta::format_time [$vertex slew fall min] $digits]:[::sta::format_time [$vertex slew fall max] $digits]"
      #set pin_slew_ "[::sta::rise_short_name] [expr {[$vertex slew rise max] - [$vertex slew rise min]}] [::sta::fall_short_name] [expr {[$vertex slew fall max] - [$vertex slew fall min]}]"
      lappend pin_slew $pin_slew_
    } else {
      set pin_slew_ "[::sta::rise_short_name] [::sta::format_time [$vertex slew_corner rise $corner min] $digits]:[::sta::format_time [$vertex slew_corner rise $corner max] $digits] [::sta::fall_short_name] [::sta::format_time [$vertex slew_corner fall $corner min] $digits]:[::sta::format_time [$vertex slew_corner fall $corner max] $digits]"
      #set pin_slew_ "[::sta::rise_short_name] [expr {[$vertex slew_corner rise $corner max] - [$vertex slew_corner rise $corner min]}] [::sta::fall_short_name] [expr {[$vertex slew_corner fall $corner max] - [$vertex slew_corner fall $corner min]}]"
      lappend pin_slew $pin_slew_    
    }
  }
  if {[llength $pin_slew] == 0} {
    lappend pin_slew "None"
  }
  return $pin_slew
}

#report_edge
proc print_ip_op_pairs {outfile input_pins output_pins is_net} {
  foreach i_p_ $input_pins {
    foreach o_p_ $output_pins {
      set input_pin [get_pin $i_p_]
      set output_pin [get_pin $o_p_]
      #::sta::report_edges -from $input_pin -to $output_pin
      foreach from_vertex [$input_pin vertices] {
        foreach to_vertex [$output_pin vertices] {
          set iter [$from_vertex out_edge_iterator]
          set arc_delays {}
          while {[$iter has_next]} {
            set edge [$iter next]
            if { [$edge to] == $to_vertex } {
              if { [$edge role] == "wire" } {
                set arc_delays_ [get_arc_delay $edge ::sta::vertex_path_name ::sta::vertex_path_name]
                lappend arc_delays $arc_delays_
              } else {
                set arc_delays_ [get_arc_delay $edge ::sta::vertex_port_name ::sta::vertex_port_name]
                lappend arc_delays $arc_delays_
              }
            }  
          }
        }
      }
      if {[llength $arc_delays] > 0} {
        set arc_delay ""
        foreach arc_delays_ $arc_delays {
          if {$arc_delay == ""} {
            set arc_delay $arc_delays_
          } else {
            set arc_delay "$arc_delay $arc_delays_"
          }
        }
      } else {
        set arc_delay " "
      }
      puts $outfile "${i_p_},${o_p_},$is_net, $arc_delay"
    }
  }
}

proc get_arc_delay {edge vertex_from_name_proc vertex_to_name_proc} {
  global sta_report_default_digits
  #set latch_enable [$edge latch_d_to_q_en]
  #if { $latch_enable != "" } {
  #  set latch_enable " enable $latch_enable"
  #}
  ##::sta::report_line "[$vertex_from_name_proc [$edge from]] -> [$vertex_to_name_proc [$edge to]] [$edge role]$latch_enable"
  set disables [::sta::edge_disable_reason $edge]
  #if { $disables != "" } {
  #  ::sta::report_line "  Disabled by $disables"
  #}
  
  set cond [$edge cond]
  #if { $cond != "" } {
  #  ::sta::report_line "  Condition: $cond"
  #}

  set mode_name [$edge mode_name]
  #if { $mode_name != "" } {
  #  ::sta::report_line "  Mode: $mode_name [$edge mode_value]"
  #}
  set arc_delay ""
  foreach arc [$edge timing_arcs] {
    set delays [$edge arc_delay_strings $arc $sta_report_default_digits]
    set delays_fmt [::sta::format_delays $delays]
    #set disable_reason ""
    #if { [::sta::timing_arc_disabled $edge $arc] } {
    #  set disable_reason " disabled"
    #}
    #::sta::report_line "  [$arc from_edge] -> [$arc to_edge] $delays_fmt$disable_reason"
    set delay_ "[$arc from_edge] -> [$arc to_edge] $delays_fmt"
    if {$arc_delay == ""} {
      set arc_delay $delay_
    } else {
      set arc_delay "$arc_delay $delay_"
    }
  }
  return $arc_delay
}

#proc get_libcell_leakage {} {
#
#}

proc load_design {def netlist libs tech_lef lefs sdc design spef} {
  foreach libFile $libs {
    read_liberty $libFile
  }
  read_lef $tech_lef
  foreach lef $lefs {
    read_lef $lef
  }
  #read_verilog $netlist
  read_def $def
  read_spef $spef
  #read_db $db
  #link_design $design
  read_sdc $sdc
  set_propagated_clock [all_clocks]
  # Ensure all OR created (rsz/cts) instances are connected
  add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
  add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
  global_connect
}

proc get_ITerm_name {ITerm} {
  set MTerm_name [[$ITerm getMTerm] getName]
  set inst_name [[$ITerm getInst] getName]
  set ITerm_name "${inst_name}/${MTerm_name}"
  return $ITerm_name
}

proc print_ip_op_cell_pairs {outfile inputs outputs} {
  foreach input $inputs {
    foreach output $outputs {
      puts $outfile "${input},${output}"
    }
  }
}

#proc print_ip_op_pairs {outfile inputs outputs is_net} {
#  foreach input $inputs {
#    foreach output $outputs {
#      puts $outfile "${input},${output},$is_net"
#    }
#  }
#}

proc report_flute_net { net } {
  set pins [lassign $net net_name drvr_index]
  puts "Net $net_name"
  set xs {}
  set ys {}
  foreach pin $pins {
    lassign $pin pin_name x y
    lappend xs $x
    lappend ys $y
  }
  stt::report_flute_tree $xs $ys 0
  #stt::report_flute_tree $xs $ys $drvr_index
}

proc print_pin_property_entry {outfile pin_props} {
  set pin_entry {}
  lappend pin_entry [dict get $pin_props "pin_name"];#pin_name 
  lappend pin_entry [dict get $pin_props "x"];#x
  lappend pin_entry [dict get $pin_props "y"];#y
  lappend pin_entry "-1";#is_port
  lappend pin_entry [dict get $pin_props "is_startpoint"];#is_startpoint
  lappend pin_entry [dict get $pin_props "is_endpoint"];#is_endpoint
  lappend pin_entry [dict get $pin_props "dir"];#dir 
  lappend pin_entry "-1";#maxcap
  lappend pin_entry [dict get $pin_props "maxtran"];#maxtran
  lappend pin_entry [dict get $pin_props "num_reachable_endpoint"];#num_reachable_endpoint
  lappend pin_entry [dict get $pin_props "cell_name"];#cell_name 
  lappend pin_entry [dict get $pin_props "net_name"];#net_name 
  lappend pin_entry [dict get $pin_props "pin_tran"];#pin_tran
  lappend pin_entry [dict get $pin_props "pin_slack"];#pin_slack
  lappend pin_entry [dict get $pin_props "pin_rise_arr"];#pin_rise_arr
  lappend pin_entry [dict get $pin_props "pin_fall_arr"];#pin_fall_arr
  lappend pin_entry [dict get $pin_props "input_pin_cap"];#input_pin_cap
  puts $outfile [join $pin_entry ","]
}

proc print_cell_property_entry {outfile cell_props} {
  set cell_entry {}
  lappend cell_entry [dict get $cell_props "cell_name"];#cell_name 
  lappend cell_entry [dict get $cell_props "is_seq"];#is_seq
  lappend cell_entry [dict get $cell_props "is_macro"];#is_macro
  #lappend cell_entry "-1";#is_in_clk
  #done is_in_clk
  lappend cell_entry [dict get $cell_props "is_in_clk"];#is_in_clk
  lappend cell_entry [dict get $cell_props "x0"];#x0
  lappend cell_entry [dict get $cell_props "y0"];#y0
  lappend cell_entry [dict get $cell_props "x1"];#x1
  lappend cell_entry [dict get $cell_props "y1"];#y1
  lappend cell_entry [dict get $cell_props "is_buf"];#is_buf
  lappend cell_entry [dict get $cell_props "is_inv"];#is_inv
  lappend cell_entry [dict get $cell_props "libcell_name"];#libcell_name 
  lappend cell_entry [dict get $cell_props "cell_static_power"];#cell_static_power
  lappend cell_entry [dict get $cell_props "cell_dynamic_power"];#cell_dynamic_power
  puts $outfile [join $cell_entry ","]
}

proc print_net_property_entry {outfile net_props} {
  set net_entry {}
  lappend net_entry [dict get $net_props "net_name"];#net_name
  lappend net_entry [dict get $net_props "net_route_length"];#net_route_length

  lappend net_entry "-1";#net_steiner_length
  lappend net_entry [dict get $net_props "fanout"];#fanout
  lappend net_entry [dict get $net_props "total_cap"];#total_cap
  lappend net_entry [dict get $net_props "net_cap"];#net_cap
  lappend net_entry [dict get $net_props "net_coupling"];#net_coupling
  lappend net_entry [dict get $net_props "net_res"];#net_res
  puts $outfile [join $net_entry ","]
}

proc print_libcell_property_entry {outfile libcell_props} {
  set libcell_entry {}
  lappend libcell_entry [dict get $libcell_props "libcell_name"];#libcell_name
  lappend libcell_entry [dict get $libcell_props "func_id"];##func. id (*8)
  lappend libcell_entry [dict get $libcell_props "libcell_area"];#libcell_area
  lappend libcell_entry "-1";#worst_input_cap(*5)
  lappend libcell_entry "-1";#libcell_leakage
  lappend libcell_entry "-1";#fo4_delay
  lappend libcell_entry "-1";#libcell_delay_fixed_load
  puts $outfile [join $libcell_entry ","]
}

