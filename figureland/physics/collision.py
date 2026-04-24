"""
Collision detection utilities.
"""

import torch
from typing import Tuple, List
from ..shapes import Shape


def detect_wall_collisions(shape: Shape, bounds: Tuple[float, float]) -> torch.Tensor:
    """Detect wall collisions for batch of shapes."""
    min_x, min_y = bounds
    max_x, max_y = bounds

    collisions = torch.zeros(shape.position.shape[0], 4, dtype=torch.bool, device=shape.device)

    collisions[:, 0] = shape.position[:, 0] - shape.size[:, 0] < min_x  # Left
    collisions[:, 1] = shape.position[:, 0] + shape.size[:, 0] > max_x  # Right
    collisions[:, 2] = shape.position[:, 1] - shape.size[:, 1] < min_y  # Bottom
    collisions[:, 3] = shape.position[:, 1] + shape.size[:, 1] > max_y  # Top

    return collisions


def detect_collisions(shapes: List[Shape]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect collisions between all pairs of shapes."""
    n_shapes = len(shapes)
    batch_size = shapes[0].position.shape[0]
    device = shapes[0].device

    collision_matrix = torch.zeros(n_shapes, n_shapes, batch_size, dtype=torch.bool, device=device)
    distances = torch.zeros(n_shapes, n_shapes, batch_size, device=device)

    for i in range(n_shapes):
        for j in range(i + 1, n_shapes):
            delta = shapes[j].position - shapes[i].position
            dist = torch.norm(delta, dim=1)
            min_dist = shapes[i].size[:, 0] + shapes[j].size[:, 0]

            collision = dist < min_dist
            collision_matrix[i, j] = collision
            collision_matrix[j, i] = collision
            distances[i, j] = dist
            distances[j, i] = dist

    return collision_matrix, distances


def resolve_collisions(shapes: List[Shape]) -> None:
    """Resolve collisions between all pairs of shapes."""
    n_shapes = len(shapes)

    for i in range(n_shapes):
        for j in range(i + 1, n_shapes):
            shape_a = shapes[i]
            shape_b = shapes[j]

            delta = shape_b.position - shape_a.position
            dist = torch.norm(delta, dim=1)
            min_dist = shape_a.size[:, 0] + shape_b.size[:, 0]
            collision = dist < min_dist

            if torch.any(collision):
                normal = delta[collision] / dist[collision].view(-1, 1)
                overlap = min_dist[collision] - dist[collision]
                total_mass = shape_a.mass[collision] + shape_b.mass[collision]

                shape_a.position[collision] -= normal * (overlap * shape_b.mass[collision] / total_mass).view(-1, 1)
                shape_b.position[collision] += normal * (overlap * shape_a.mass[collision] / total_mass).view(-1, 1)

                rel_vel = shape_b.velocity[collision] - shape_a.velocity[collision]
                vel_along_normal = torch.sum(rel_vel * normal, dim=1)

                approaching = vel_along_normal < 0
                if torch.any(approaching):
                    e = torch.minimum(shape_a.elasticity[collision][approaching], shape_b.elasticity[collision][approaching])
                    j = -(1 + e) * vel_along_normal[approaching]
                    j /= 1.0 / shape_a.mass[collision][approaching] + 1.0 / shape_b.mass[collision][approaching]

                    shape_a.velocity[collision][approaching] -= normal[approaching] * j.view(-1, 1) / shape_a.mass[collision][approaching].view(-1, 1)
                    shape_b.velocity[collision][approaching] += normal[approaching] * j.view(-1, 1) / shape_b.mass[collision][approaching].view(-1, 1)
