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



import re
import dgl
import torch
import pandas as pd
import numpy as np
from numpy.random import *
import pickle
import sys
from helper import (
    get_large_components,
    get_subgraph,
    get_cell_graph_from_cells,
)
from generate_LPG_from_tables import generate_LPG_from_tables
from pdb import set_trace as bp

### extract buffer trees from netlist ###
### inputs: data_root, design name
### output:
### nodes: celll/pin id, v_tree_id, v_BT_height, v_bt_s, v_x, v_y, v_arr, v_tran, v_polarity, v_libcell_id
### edges: src, tar, e_tree_id
def load_sampling(data_root):
    g, pin_df, cell_df, net_df, fo4_df, pin_pin_df, cell_pin_df, \
        net_pin_df, net_cell_df, cell_cell_df, edge_df, v_type, e_type \
        = generate_LPG_from_tables(data_root = data_root)

    ### get dimensions
    num_cells = cell_df.shape
    num_nets = net_df.shape
    num_pins = pin_df.shape
    total_v_cnt = num_pins
    total_e_cnt = num_cells + num_nets
    
    pin_df['node_id'] = pin_df.index
    cell_df['cell_id'] = cell_df.index
    
    bp()
    
    edge_index = pin_pin_df.loc[((pin_pin_df.src.isin(pin_df.name)) & (pin_pin_df.tar.isin(pin_df.name)))]
    edges_df = pin_pin_df.loc[edge_index.index]
    pin_id_df = pin_df[['name', 'node_id']].set_index('name')
    edges_df['src_id'] = pin_id_df.loc[edges_df['src'].tolist()].node_id.tolist()
    edges_df['tar_id'] = pin_id_df.loc[edges_df['tar'].tolist()].node_id.tolist()
    net_id_df = net_df[['name', 'id']].set_index('name')
    cell_id_df = cell_df[['name', 'cell_id']].set_index('name')
    
    net_edge_index = edge_index.loc[edge_index.is_net == 1].copy()
    cell_edge_index = edge_index.loc[edge_index.is_net == 0].copy()
    assert(net_edge_index.shape[0] + cell_edge_index.shape[0] == edge_index.shape[0])

    pin_net_edge_index = net_pin_df.loc[net_pin_df.src_type == 'pin'].set_index('src_id')
    pin_cell_edge_index = cell_pin_df.loc[cell_pin_df.tar_type == 'pin'].set_index('tar_id')
    
    net_edge_index['net_id'] = pin_net_edge_index.loc[net_edge_index['src_id'].tolist()]['tar_id'].tolist()
    cell_edge_index['cell_id'] = pin_cell_edge_index.loc[cell_edge_index['tar_id'].tolist()]['src_id'].tolist()
    
    pins_net = [torch.tensor(net_edge_index.src_id.tolist()), torch.tensor(net_edge_index.tar_id.tolist())]
    pins_cell = [torch.tensor(cell_edge_index.src_id.tolist()), torch.tensor(cell_edge_index.tar_id.tolist())]
    
    net_edge_id = (net_edge_index.net_id - num_cells[0] - num_pins[0]).tolist()
    cell_edge_id = (cell_edge_index.cell_id - num_pins[0]).tolist()
    
    pin_property = torch.tensor(pin_df[['name', 'x', 'y', 'is_in_clk', 'is_inv']].to_numpy())
    net_property = torch.tensor(net_df.loc[net_edge_id][['fanout', 'total_cap']].to_numpy())
    cell_property = torch.tensor(cell_df.loc[cell_edge_id][['x0', 'y0', 'x1', 'y1', 'fix_load_delay', 'size_cnt']].to_numpy())
                                
    graph_data = {("pin", "net", "pin"): (pins_net[0], pins_net[1]), ("pin", "cell", "pin"): (pins_cell[0], pins_cell[1])}
    g = dgl.heterograph(graph_data)
    
    g.nodes["pin"].data["pf"] = pin_property
    g.edges["net"].data["nf"] = net_property
    g.edges["cell"].data["cf"] = cell_property
    
if __name__ == "__main__":
    data_root = sys.argv[1]
    output_path = sys.argv[2]
    g = load_sampling(data_root)
    
    # save graph
    dgl.save_graphs("{}/{}.pth".format(output_path, "graph"), g)
    
    print("------------Done------------")
