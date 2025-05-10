import torch
import torch.nn as nn
from torch_geometric.data import Data
import random


class PromptEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=32, prompt_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, prompt_dim)
        )

    def forward(self, prompt_ids):
        embedded = self.embedding(prompt_ids).mean(dim=1)
        return self.fc(embedded)


class GraphGenerator(nn.Module):
    def __init__(self, prompt_dim, hidden_dim=64, num_nodes=20):
        super().__init__()
        self.num_nodes = num_nodes
        self.fc = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * 3 + num_nodes * num_nodes)
        )

    def forward(self, prompt_embedding):
        num_rooms = 2  # or parse from prompt
        room_size = 10.0
        room_spacing = 20.0
        rooms = []

        node_feats = []
        node_types = []
        edge_index = []

        room_boxes = {}

        # Place rooms on a grid
        for i in range(num_rooms):
            x = (i % 2) * room_spacing
            y = 0
            z = (i // 2) * room_spacing
            rooms.append((x, y, z))

            # Room node
            node_feats.append([x, y, z])
            node_types.append(0)

            # Save box
            room_boxes[i] = {
                "center": (x, y, z),
                "size": room_size,
            }

        # Add spawns inside rooms
        for i, (rx, ry, rz) in enumerate(rooms):
            sx = rx + (torch.rand(1).item() - 0.5) * room_size * 0.5
            sy = ry
            sz = rz + (torch.rand(1).item() - 0.5) * room_size * 0.5
            node_feats.append([sx, sy, sz])
            node_types.append(1)
            edge_index.append([i, len(node_feats)-1])  # room → spawn

        # Add doors between adjacent rooms
        for i in range(num_rooms - 1):
            x1, y1, z1 = rooms[i]
            x2, y2, z2 = rooms[i+1]
            dx = (x1 + x2) / 2
            dy = (y1 + y2) / 2
            dz = (z1 + z2) / 2
            node_feats.append([dx, dy, dz])
            node_types.append(2)
            door_idx = len(node_feats) - 1
            edge_index.append([i, door_idx])
            edge_index.append([i+1, door_idx])

        # Finalize
        x = torch.tensor(node_feats, dtype=torch.float)
        node_type = torch.tensor(node_types, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index, node_type=node_type, room_boxes=room_boxes)
