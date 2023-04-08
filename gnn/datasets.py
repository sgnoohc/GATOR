import torch
from torch.utils.data import Dataset

class EdgeData:
    def __init__(self, node_attr, edge_idxs, edge_attr, labels):
        self.x = node_attr
        self.edge_index = edge_idxs
        self.edge_attr = edge_attr
        self.y = labels

    def to(self, device):
        self.x.to(device)
        self.edge_attr.to(device)
        self.y.to(device)
        return self

class EdgeDataset(Dataset):
    def __init__(self, pyg_data):
        self.data = pyg_data
        idx_ranges = []
        for event_i in range(len(self.data)):
            if event_i == 0:
                idx_ranges.append((0, self.data[event_i].num_edges - 1))
            else:
                n = idx_ranges[-1][-1]
                idx_ranges.append((n + 1, n + self.data[event_i].num_edges - 1))

        self.__idx_range = self.__range_lims(idx_ranges)
        self.__idx_ranges = {}
        for i, r in enumerate(idx_ranges):
            self.__idx_ranges[r] = i

        self.__idx_map = self.__split_idx_map(idx_ranges)

    def __range_lims(self, idx_ranges):
        return (idx_ranges[0][0], idx_ranges[-1][-1])

    def __split_idx_map(self, idx_ranges):
        splt = len(idx_ranges)//2
        left = idx_ranges[:splt]
        right = idx_ranges[splt:]
        if splt <= 2:
            splt_idx_map = {
                self.__range_lims(left): left,
                self.__range_lims(right): right
            }
            return splt_idx_map
        else:
            return {
                self.__range_lims(idx_ranges): {
                    self.__range_lims(left): self.__split_idx_map(left),
                    self.__range_lims(right): self.__split_idx_map(right)
                }
            }

    def __find_idx(self, idx):
        finished = True
        next_search = self.__idx_map[self.__idx_range]
        while type(next_search) == dict:
            found = False
            for key in next_search.keys():
                low, high = key
                if low <= idx and idx <= high:
                    next_search = next_search[key]
                    found = True
                    continue

            if not found:
                finished = False
                break

        if finished:
            found = False
            for idx_range in next_search:
                low, high = idx_range
                if low <= idx and idx <= high:
                    found = True
                    return idx_range, self.__idx_ranges[idx_range]

            if not found:
                raise Exception("this should never happen!")
        else:
            raise Exception(f"{idx} is out of range {self.__idx_range}")

    def __len__(self):
        return self.__idx_range[-1]+1

    def __getitem__(self, idx):
        (low, high), event = self.__find_idx(idx)
        data = self.data[event]
        edge_attr = data.edge_attr[idx - low]
        i, j = data.edge_index[:,idx - low]
        node_i = data.x[i].unsqueeze(1)
        node_j = data.x[j].unsqueeze(1)
        node_attr = torch.cat((node_i, node_j), dim=1)
        return EdgeData(
            torch.transpose(node_attr, 0, 1),
            (i, j),
            edge_attr,
            data.y[idx - low].unsqueeze(1)
        )
