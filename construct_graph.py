import os


os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from tqdm import tqdm




def construct_graph(cellMap, NetMap, cells, nets, pins):
    ## construct heterogeneous graph
    pin_u_raw = [[], []]
    pin_v_raw = [[], []]
    pin_slack_rise_raw = [[], []]
    pin_slack_fall_raw = [[], []]
    pin_x_raw = [[], []]
    pin_y_raw = [[], []]


    num_cells = len(cells)
    num_nets = len(nets)
    num_pins = len(pins)


    ## Feature metrics
    """
    @param: c_x, c_y: cell positions
    @param: c_touch: cell is dont_touch
    @param: n_touch: net is dont_touch
    @param: p_tie: pin is connected to TIE nets
    @param: p_slack_: pin slacks
    @param: n_num_loads: net #loads
    """
    c_touch = torch.zeros([num_cells, 1], dtype=torch.float32)
    c_delay = torch.zeros([num_cells, 1], dtype=torch.float32)
    n_touch = torch.zeros([num_nets, 1], dtype=torch.float32)
    n_delay = torch.zeros([num_nets, 1], dtype=torch.float32)
    n_length = torch.zeros([num_nets, 1], dtype=torch.float32)
    n_num_loads = torch.zeros([num_nets, 1], dtype=torch.float32)
    n_is_tie = torch.zeros([num_nets, 1], dtype=torch.float32)
    c_names = []


    ## load graph and net/pin features
    for net in tqdm(nets):
        # n_index = net.id + num_cells # FIXME: should add num_cells?
        n_index = net.id
        for pin_id in net.pins:
            pin = pins[pin_id]
            c_index = pin.cell_id
            dir = pin.dir
            if dir == 0:
                pin_u_raw[0].append(n_index)
                pin_v_raw[0].append(c_index)
            else:
                pin_u_raw[1].append(c_index)
                pin_v_raw[1].append(n_index)
            pin_slack_rise_raw[dir].append(pin.slack_rise if (pin.slack_rise != float("inf")) else 0)
            pin_slack_fall_raw[dir].append(pin.slack_fall if (pin.slack_fall != float("inf")) else 0)
            pin_x_raw[dir].append(pin.x)
            pin_y_raw[dir].append(pin.y)


        n_touch[n_index] = float(net.is_dont_touch)
        n_is_tie[n_index] = float(net.is_tie)
        n_num_loads[n_index] = float(net.num_loads)
        n_length[n_index] = float(net.length)
        n_delay[n_index] = float(net.delay)


    ## load graph and cell features
    for cell in tqdm(cells):
        c_index = cell.id
        c_touch[c_index] = float(cell.is_dont_touch)
        c_delay[c_index] = float(cell.delay)


    ## process pin features from raw data
    p_u = [torch.tensor(pin_u_raw[0]), torch.tensor(pin_u_raw[1])]
    p_v = [torch.tensor(pin_v_raw[0]), torch.tensor(pin_v_raw[1])]
    p_slack_rise = [
        torch.tensor(pin_slack_rise_raw[0]).unsqueeze(1),
        torch.tensor(pin_slack_rise_raw[1]).unsqueeze(1),
    ]
    p_slack_fall = [
        torch.tensor(pin_slack_fall_raw[0]).unsqueeze(1),
        torch.tensor(pin_slack_fall_raw[1]).unsqueeze(1),
    ]
    p_x = [
        torch.tensor(pin_x_raw[0]).unsqueeze(1),
        torch.tensor(pin_x_raw[1]).unsqueeze(1),
    ]
    p_y = [
        torch.tensor(pin_y_raw[0]).unsqueeze(1),
        torch.tensor(pin_y_raw[1]).unsqueeze(1),
    ]


    ## construct dgl graoh data
    num_nodes_dict = {"net": num_nets, "inst": num_cells}
    graph_data = {
        ("net", "pin_in", "inst"): (p_u[0], p_v[0]),
        ("inst", "pin_out", "net"): (p_u[1], p_v[1]),
    }


    ## construct dgl graph with edge/node features
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    g.nodes["inst"].data["cf"] = torch.cat([c_touch, c_delay], 1)
    g.nodes["net"].data["nf"] = torch.cat([n_touch, n_num_loads, n_length, n_delay], 1)
    etypes = ["pin_in", "pin_out"]
    for etype in etypes:
        g.edges[etype].data["ef"] = torch.cat(
            [
                p_slack_rise[etypes.index(etype)],
                p_slack_fall[etypes.index(etype)],
                p_x[etypes.index(etype)],
                p_y[etypes.index(etype)],
            ],
            1,
        )
    return g


