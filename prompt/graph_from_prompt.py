import torch
from graph_generator import PromptEncoder, GraphGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dummy prompt string
prompt = "Generate a simple map with 20 rooms and a door between them. The rooms are touching on one side."

# Dummy tokenizer: convert chars to ints (you can replace this with real token IDs later)
prompt_ids = torch.tensor([[ord(c) % 1000 for c in prompt]], dtype=torch.long)  # (1, sequence_len)

# Initialize models
encoder = PromptEncoder()
generator = GraphGenerator(prompt_dim=128)

# Forward pass
prompt_embed = encoder(prompt_ids)
graph = generator(prompt_embed)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
colors = {0: 'blue', 1: 'green', 2: 'red'}  # room, spawn, door
for i, (x, y, z) in enumerate(graph.x.tolist()):
    node_type = graph.node_type[i].item()
    ax.scatter(x, y, z, c=colors[node_type], label=f"Node {i}" if i == 0 else "", s=50)

# Plot edges
for i, j in graph.edge_index.t().tolist():
    x_coords = [graph.x[i][0].item(), graph.x[j][0].item()]
    y_coords = [graph.x[i][1].item(), graph.x[j][1].item()]
    z_coords = [graph.x[i][2].item(), graph.x[j][2].item()]
    ax.plot(x_coords, y_coords, z_coords, c='black', linewidth=0.5)

# Plot room boxes
for room_id, box in graph.room_boxes.items():
    cx, cy, cz = box['center']
    size = box['size']
    r = size / 2
    for s, e in [
        # 4 bottom square edges
        ((-r, -r, -r), (r, -r, -r)),
        ((r, -r, -r), (r, r, -r)),
        ((r, r, -r), (-r, r, -r)),
        ((-r, r, -r), (-r, -r, -r)),
        # 4 verticals
        ((-r, -r, -r), (-r, -r, r)),
        ((r, -r, -r), (r, -r, r)),
        ((r, r, -r), (r, r, r)),
        ((-r, r, -r), (-r, r, r)),
        # 4 top square edges
        ((-r, -r, r), (r, -r, r)),
        ((r, -r, r), (r, r, r)),
        ((r, r, r), (-r, r, r)),
        ((-r, r, r), (-r, -r, r))
    ]:
        sx, sy, sz = [cx + s[d] for d in range(3)]
        ex, ey, ez = [cx + e[d] for d in range(3)]
        ax.plot([sx, ex], [sy, ey], [sz, ez], color='cyan', linewidth=0.8)

ax.set_title("Generated Map Graph")
plt.show()
