import dgl
import torch
from .data_graph import *
import dgl.function as fn
import gc




def clean_object_from_memory(obj):  # definition
    del obj
    gc.collect()
    torch.cuda.empty_cache()




def index2vector(num, index):
    vector = torch.zeros(num, dtype=torch.float32)
    vector[index] = 1
    return vector




def process_subgraph_index(num, sub_net_cell, g):
    sub_net_cell_vec = index2vector(num, sub_net_cell)
    parent_id = g.ndata[dgl.NID]["inst"]
    parent_map_to_sub = torch.zeros(num, dtype=torch.long)
    position_index = torch.arange(len(sub_net_cell))
    parent_map_to_sub[parent_id] = position_index
    return sub_net_cell_vec, parent_map_to_sub




def preprocess_graph_info(g, atom, args, logger):
    if not atom["preprocessed"]:
        ## propogate to instance slack
        def calc_inst_slack(edges):
            slack = torch.min(edges.data["ef"][:, 0], edges.data["ef"][:, 1])
            return {"e_slack": slack}


        g.update_all(calc_inst_slack, fn.min("e_slack", "n_slack"), etype="pin_in")
        in_slack = (g.nodes["inst"].data["n_slack"] > 0).to(torch.float32).unsqueeze(1)
        logger.info("In Positive slack #inst: %d" % torch.sum(in_slack))


        ## propogate to instance fanout
        def calc_inst_fanout(edges):
            x = edges.src["nf"][:, 1]
            return {"e_fanout": x}


        ## reverse graph and calculate instance fanout
        rg = dgl.reverse(g, copy_edata=True)
        rg.update_all(calc_inst_fanout, fn.sum("e_fanout", "i_fanout"), etype="pin_out")
        # g.nodes["inst"].data["i_fanout"] = rg.nodes["inst"].data["i_fanout"]
        rg.nodes["inst"].data["single_fanout_inst"] = (
            (rg.nodes["inst"].data["i_fanout"] <= 1).to(torch.float32).unsqueeze(1)
        )
        rg.update_all(calc_inst_slack, fn.min("e_slack", "n_slack"), etype="pin_out")
        out_slack = (rg.nodes["inst"].data["n_slack"] > 0).to(torch.float32).unsqueeze(1)
        logger.info("Out Positive slack #inst: %d" % torch.sum(out_slack))


        rg.nodes["inst"].data["positive_slack_inst"] = ((in_slack + out_slack) >= 1).to(torch.float32)


        logger.info("Positive slack #inst: %d" % torch.sum(rg.nodes["inst"].data["positive_slack_inst"]))


        g = rg
        clean_object_from_memory(rg)


        ## FIXME: whether to cat feature on insts
        g.nodes["inst"].data["cf"] = torch.cat(
            [
                g.nodes["inst"].data["cf"],
                g.nodes["inst"].data["n_slack"].unsqueeze(1),
                g.nodes["inst"].data["i_fanout"].unsqueeze(1),
            ],
            dim=1,
        )
        ## process insts slack
        g.nodes["inst"].data["cf"][:, 2].clamp_(None, 0.0)
        # g.nodes["inst"].data["cf"][:, 2] *= -1 * args.scaler  ## input inst_slack
        # g.nodes["inst"].data["cf"][:, 3] *= args.scaler  ## num_loads
        g.nodes["inst"].data["cf"][:, 2] /= g.nodes["inst"].data["cf"][:, 2].mean()
        g.nodes["inst"].data["cf"][:, 3] /= g.nodes["inst"].data["cf"][:, 3].mean()


        ## TODO: whether to clamp slack
        g.edges["pin_in"].data["ef"][:, :2].clamp_(None, 1.0)
        g.edges["pin_out"].data["ef"][:, :2].clamp_(None, 1.0)


        # ## FIXME: whether to process input pin slack
        # g.edges["pin_in"].data["ef"][:, :2] *= -1 * args.scaler  ## input pin_slack
        # g.edges["pin_out"].data["ef"][:, :2] *= -1 * args.scaler  ## output pin_slack


        g.edges["pin_in"].data["ef"] *= -1
        g.edges["pin_out"].data["ef"] *= -1
        g.edges["pin_in"].data["ef"] = torch.nn.functional.normalize(g.edges["pin_in"].data["ef"]) * args.scaler
        g.edges["pin_out"].data["ef"] = torch.nn.functional.normalize(g.edges["pin_out"].data["ef"]) * args.scaler


        ## FIXME: whether to process net length
        g.edges["pin_in"].data["ef"][:, 2:4] /= g.edges["pin_in"].data["ef"][:, 2:4].mean()
        # g.edges["pin_in"].data["ef"][:, 3] /= g.edges["pin_in"].data["ef"][:, 3].mean()
        g.edges["pin_out"].data["ef"][:, 2:4] /= g.edges["pin_out"].data["ef"][:, 2:4].mean()
        # g.edges["pin_out"].data["ef"][:, 3] /= g.edges["pin_out"].data["ef"][:, 3].mean()


        ## FIXME: process binary values
        g.nodes["inst"].data["cf"][:, 0] = 1 - g.nodes["inst"].data["cf"][:, 0]  ## dont_touch
        g.nodes["net"].data["nf"][:, 0] = 1 - g.nodes["net"].data["nf"][:, 0]  ## dont_touch


        # ## FIXME: process #fanout
        # g.nodes["net"].data["nf"][:, 1] *= args.scaler  ## num_loads


        ## FIXME: process delays
        g.nodes["net"].data["nf"][:, 3].clamp_(0.0, None)
        g.nodes["inst"].data["cf"][:, 1].clamp_(0.0, None)


        g.nodes["net"].data["nf"][:, 2] /= g.nodes["net"].data["nf"][:, 2].mean()
        g.nodes["net"].data["nf"][:, 3] /= g.nodes["net"].data["nf"][:, 3].mean()
        g.nodes["inst"].data["cf"][:, 1] /= g.nodes["inst"].data["cf"][:, 1].mean()


        ## end preprocess
        atom["preprocessed"] = True
        return g




def report_slack_weight(g, model, logger):
    # def calc_inst_slack(edges):
    #     slack = torch.min(edges.data["ef"][:, 0], edges.data["ef"][:, 1])
    #     return {"e_slack": slack}


    # g.update_all(calc_inst_slack, fn.min("e_slack", "n_slack"), etype="pin_in")


    # n_slack = g.nodes["inst"].data["n_slack"]
    positive_slack_inst = g.nodes["inst"].data["positive_slack_inst"].to(torch.float32)
    negative_slack_inst = (g.nodes["inst"].data["positive_slack_inst"] == 0).to(torch.float32)


    num_negative_slack_inst = torch.sum(1 - positive_slack_inst).item()
    num_postive_slack_inst = torch.sum(positive_slack_inst).item()


    pred_inst_tat = model(g, False)


    negative_slack_inst_weight = pred_inst_tat * negative_slack_inst
    positive_slack_inst_weight = pred_inst_tat * positive_slack_inst


    logger.info(
        "Negative/Positive slack #inst: %d/%d, avg weight: %.5f/%.5f"
        % (
            num_negative_slack_inst,
            num_postive_slack_inst,
            torch.sum(negative_slack_inst_weight) / num_negative_slack_inst,
            torch.sum(positive_slack_inst_weight) / num_postive_slack_inst,
        )
    )
    # logger.info(
    #     "Positive slack #inst: %d, avg weight: %.5f"
    #     % (num_postive_slack_inst, torch.sum(positive_slack_inst_weight) / num_postive_slack_inst)
    # )




class positive_slack_weight_penalty(torch.nn.Module):
    def __init__(self):
        super(positive_slack_weight_penalty, self).__init__()


    def forward(self, y_pred, positive_slack_inst):
        # return torch.sum(y_pred.abs() * positive_slack_inst) + torch.sum((y_pred - 1).abs() * (1 - positive_slack_inst))
        return torch.sum(y_pred.abs() * positive_slack_inst)




class single_fanout_weight_penalty(torch.nn.Module):
    def __init__(self):
        super(single_fanout_weight_penalty, self).__init__()


    def forward(self, y_pred, single_fanout_inst):
        # return torch.sum(y_pred.abs() * positive_slack_inst) + torch.sum((y_pred - 1).abs() * (1 - positive_slack_inst))
        return torch.sum(y_pred.abs() * single_fanout_inst)




class negative_weight_penalty(torch.nn.Module):
    def __init__(self):
        super(negative_weight_penalty, self).__init__()


    def forward(self, y_pred):
        return torch.sum(torch.nn.ReLU()(-y_pred))
