"""
Physics engine implementation with pure PyTorch operations.
Supports batched simulation of multiple environments simultaneously.
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any, Union, List
from ..shapes import Shape
from ..exceptions import (
    ShapeOutOfBoundsError,
    ShapePenetrationError,
    DeviceMismatchError,
    BatchSizeMismatchError,
    EmptyEnvironmentError
)


class Environment:
    """
    Environment container that manages shapes, boundaries, and force fields.

    Follows proper OOP encapsulation:
    - Shapes are owned and managed by the Environment
    - Automatic validation on shape addition
    - Penetration and bounds checking
    - Proper error handling
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        environment_type: str = 'rectangle',
        gravity: float = 9.8,
        friction: float = 0.05,
        air_resistance: float = 0.01,
        force_field: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.log = logging.getLogger(__name__)
        self.device = device or torch.device('cpu')
        self.bounds = bounds
        self.environment_type = environment_type
        self.gravity = torch.tensor(gravity, device=self.device)
        self.friction = torch.tensor(friction, device=self.device)
        self.air_resistance = torch.tensor(air_resistance, device=self.device)
        self.force_field = force_field

        self.min_x, self.min_y = bounds[0], bounds[0]
        self.max_x, self.max_y = bounds[1], bounds[1]

        if environment_type == 'circle':
            self.radius = (bounds[1] - bounds[0]) / 2.0
            self.center = torch.tensor([0.0, 0.0], device=self.device)

        # Managed shapes collection
        self.shapes: List[Shape] = []

    def get_force_vector(self, position: torch.Tensor, dt: float) -> torch.Tensor:
        """Calculate force vector for given position in batch."""
        batch_size = position.shape[0]
        force = torch.zeros_like(position)

        # Gravity (downward force)
        force[:, 1] -= self.gravity

        # Air resistance
        force -= self.air_resistance * position

        if self.force_field == 'turbulence':
            noise = torch.randn_like(position) * 0.1
            force += noise

        return force

    def add_shape(self, shape: Shape, strict: bool = True) -> None:
        """
        Add a shape to the environment.

        Args:
            shape: Shape object to add
            strict: If True (default), perform full validation (device, bounds, penetration).
                    If False, skip validation and add directly (used for internal generation).
        """
        self.log.debug(f"Adding shape to environment: {type(shape).__name__} (strict={strict})")

        if strict:
            # Check device matches
            if shape.device != self.device:
                raise DeviceMismatchError(
                    f"Shape device {shape.device} does not match environment device {self.device}. "
                    f"Use shape.to({self.device}) first."
                )

            # Check batch size consistency
            if self.shapes:
                if shape.position.shape[0] != self.shapes[0].position.shape[0]:
                    raise BatchSizeMismatchError(
                        f"Shape batch size {shape.position.shape[0]} does not match "
                        f"existing batch size {self.shapes[0].position.shape[0]}"
                    )

            # Check shape is within environment bounds
            if not self._is_shape_in_bounds(shape):
                raise ShapeOutOfBoundsError(
                    f"Shape is outside environment bounds {self.bounds}. "
                    f"Position: {shape.position.mean(dim=0).numpy()}, "
                    f"Size: {shape.size.mean(dim=0).numpy()}"
                )

            # Check for penetration with existing shapes
            penetrating = self._check_penetration(shape)
            if penetrating.any():
                penetrating_count = penetrating.sum().item()
                raise ShapePenetrationError(
                    f"Shape penetrates {penetrating_count} existing shapes. "
                    "Adjust position or size to avoid overlap before adding."
                )

        # All checks passed (or skipped)
        self.shapes.append(shape)
        self.log.debug(f"Successfully added shape, total shapes: {len(self.shapes)}")

    def add_shapes(self, shapes: List[Shape], strict: bool = True) -> None:
        """Add multiple shapes at once with validation."""
        for shape in shapes:
            self.add_shape(shape, strict=strict)

    def remove_shape(self, index: int) -> Shape:
        """Remove shape at given index."""
        if 0 <= index < len(self.shapes):
            shape = self.shapes.pop(index)
            self.log.debug(f"Removed shape at index {index}")
            return shape
        raise IndexError(f"Shape index {index} out of range (0-{len(self.shapes)-1})")

    def clear_shapes(self) -> None:
        """Remove all shapes from environment."""
        self.shapes.clear()
        self.log.debug("Cleared all shapes from environment")

    def _is_shape_in_bounds(self, shape: Shape) -> torch.Tensor:
        """Check if shape is entirely within environment bounds."""
        half_size = shape.size / 2

        if self.environment_type in ['rectangle', 'square']:
            in_bounds = torch.ones(shape.position.shape[0], dtype=torch.bool, device=self.device)
            in_bounds &= (shape.position[:, 0] - half_size[:, 0]) >= self.min_x
            in_bounds &= (shape.position[:, 0] + half_size[:, 0]) <= self.max_x
            in_bounds &= (shape.position[:, 1] - half_size[:, 1]) >= self.min_y
            in_bounds &= (shape.position[:, 1] + half_size[:, 1]) <= self.max_y
            return in_bounds.all()

        elif self.environment_type == 'circle':
            pos_dist = torch.norm(shape.position - self.center, dim=1)
            in_bounds = (pos_dist + shape.size[:, 0]) <= self.radius
            return in_bounds.all()

        return True

    def _check_penetration(self, shape: Shape) -> torch.Tensor:
        """Check if new shape penetrates any existing shapes."""
        if not self.shapes:
            return torch.zeros(shape.position.shape[0], dtype=torch.bool, device=self.device)

        penetrating = torch.zeros(shape.position.shape[0], dtype=torch.bool, device=self.device)

        for existing in self.shapes:
            delta = shape.position - existing.position
            dist = torch.norm(delta, dim=1)
            min_dist = shape.size[:, 0] + existing.size[:, 0]  # sum of half-sizes
            penetrating |= dist < min_dist

        return penetrating

    def validate_state(self) -> None:
        """
        Validate entire environment state.

        Checks:
        - All shapes are within bounds
        - No shapes are penetrating each other
        - All tensors are on correct device

        Raises:
            RuntimeError: If any state is invalid
        """
        self.log.debug("Validating environment state")

        for i, shape in enumerate(self.shapes):
            if shape.device != self.device:
                raise RuntimeError(f"Shape {i} on wrong device: {shape.device} != {self.device}")

            if not self._is_shape_in_bounds(shape):
                out_count = (~self._is_shape_in_bounds(shape)).sum().item()
                if out_count > 0:
                    self.log.warning(f"Shape {i} has {out_count} instances out of bounds")

        # Check all pairs for penetration
        for i in range(len(self.shapes)):
            for j in range(i + 1, len(self.shapes)):
                delta = self.shapes[i].position - self.shapes[j].position
                dist = torch.norm(delta, dim=1)
                min_dist = (self.shapes[i].size[:, 0] + self.shapes[j].size[:, 0]) / 2.0
                penetrating = dist < min_dist

                if penetrating.any():
                    count = penetrating.sum().item()
                    self.log.warning(f"Shapes {i} and {j} are penetrating in {count} batch instances")



    def to(self, device: torch.device) -> 'Environment':
        """Move environment to specified device."""
        env = Environment(
            self.bounds,
            self.environment_type,
            float(self.gravity),
            float(self.friction),
            float(self.air_resistance),
            self.force_field,
            device=device
        )

        # Move all shapes to new device
        env.shapes = [shape.to(device) for shape in self.shapes]
        return env


    """
    Batched physics simulation engine using PyTorch tensors.

    Operates exclusively on shapes managed by the Environment.
    """

    def __init__(
        self,
        environment: 'Environment',
        dt: float = 1.0 / 60.0,
        substeps: int = 4,
        collisions_enabled: bool = True
    ):
        self.environment = environment
        self.device = environment.device
        self.log = environment.log
        self.dt = dt
        self.substeps = substeps
        self.sub_dt = dt / substeps
        self.collisions_enabled = collisions_enabled

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance simulation by one timestep with substepping.

        Uses shapes managed by the Environment.

        Returns:
            Tuple of (positions, collision_info)
        """
        shapes = self.environment.shapes

        if not shapes:
            import logging
            logging.getLogger(__name__).warning("Physics step called with no shapes in environment")
            return torch.empty(0), torch.empty(0)

        for _ in range(self.substeps):
            for shape in shapes:
                force = self.environment.get_force_vector(shape.position, self.sub_dt)
                acceleration = force / shape.mass.view(-1, 1)
                shape.velocity += acceleration * self.sub_dt
                shape.velocity *= (1.0 - self.environment.friction)
                shape.position += shape.velocity * self.sub_dt

            if self.collisions_enabled:
                for shape in shapes:
                    self._handle_wall_collisions(shape)

                if len(shapes) > 1:
                    self._handle_object_collisions(shapes)

        collision_info = torch.zeros(len(shapes), len(shapes), dtype=torch.bool, device=self.device)
        positions = torch.cat([s.position for s in shapes], dim=1)
        return positions, collision_info
    def _handle_wall_collisions(self, shape: Shape) -> None:
        """Handle wall collisions for a shape."""
        if self.environment.environment_type in ['rectangle', 'square']:
            # Left wall
            left_collision = shape.position[:, 0] - shape.size[:, 0] < self.environment.min_x
            shape.position[left_collision, 0] = self.environment.min_x + shape.size[left_collision, 0]
            shape.velocity[left_collision, 0] *= -shape.elasticity[left_collision]

            # Right wall
            right_collision = shape.position[:, 0] + shape.size[:, 0] > self.environment.max_x
            shape.position[right_collision, 0] = self.environment.max_x - shape.size[right_collision, 0]
            shape.velocity[right_collision, 0] *= -shape.elasticity[right_collision]

            # Bottom wall
            bottom_collision = shape.position[:, 1] - shape.size[:, 1] < self.environment.min_y
            shape.position[bottom_collision, 1] = self.environment.min_y + shape.size[bottom_collision, 1]
            shape.velocity[bottom_collision, 1] *= -shape.elasticity[bottom_collision]

            # Top wall
            top_collision = shape.position[:, 1] + shape.size[:, 1] > self.environment.max_y
            shape.position[top_collision, 1] = self.environment.max_y - shape.size[top_collision, 1]
            shape.velocity[top_collision, 1] *= -shape.elasticity[top_collision]

        elif self.environment.environment_type == 'circle':
            pos_norm = torch.norm(shape.position - self.environment.center, dim=1)
            circle_collision = pos_norm + shape.size[:, 0] > self.environment.radius

            if torch.any(circle_collision):
                normal = shape.position[circle_collision] / pos_norm[circle_collision].view(-1, 1)
                shape.position[circle_collision] = normal * (self.environment.radius - shape.size[circle_collision, 0])

                vel_dot = torch.sum(shape.velocity[circle_collision] * normal, dim=1, keepdim=True)
                shape.velocity[circle_collision] -= 2 * vel_dot * normal
                shape.velocity[circle_collision] *= shape.elasticity[circle_collision].view(-1, 1)



    def _handle_object_collisions(self, shapes: List[Shape]) -> None:
        """Handle object-object collisions between all pairs of shapes."""
        n_shapes = len(shapes)
        batch_size = shapes[0].position.shape[0]

        for i in range(n_shapes):
            for j in range(i + 1, n_shapes):
                shape_a = shapes[i]
                shape_b = shapes[j]

                # Vector between centers
                delta = shape_b.position - shape_a.position
                dist = torch.norm(delta, dim=1)

                # Minimum distance for collision (sum of half-sizes)
                min_dist = shape_a.size[:, 0] + shape_b.size[:, 0]
                collision = dist < min_dist

                if torch.any(collision):
                    colliding_idx = torch.where(collision)[0]

                    # Normalized collision normals for colliding pairs
                    normal = delta[collision] / dist[collision].view(-1, 1)

                    # Separate objects proportionally by inverse mass
                    overlap = min_dist[collision] - dist[collision]
                    total_mass = shape_a.mass[collision] + shape_b.mass[collision]

                    shape_a.position[collision] -= normal * (overlap * shape_b.mass[collision] / total_mass).view(-1, 1)
                    shape_b.position[collision] += normal * (overlap * shape_a.mass[collision] / total_mass).view(-1, 1)

                    # Relative velocity for colliding pairs
                    rel_vel = shape_b.velocity[collision] - shape_a.velocity[collision]
                    vel_along_normal = torch.sum(rel_vel * normal, dim=1)

                    # Only resolve if objects are moving towards each other
                    approaching = vel_along_normal < 0
                    if torch.any(approaching):
                        # Indices where both collision and approaching are true
                        update_indices = colliding_idx[approaching]

                        # Coefficient of restitution for these pairs
                        e = torch.minimum(
                            shape_a.elasticity[collision][approaching],
                            shape_b.elasticity[collision][approaching]
                        )

                        # Compute impulse magnitude
                        j = -(1 + e) * vel_along_normal[approaching]
                        inv_mass_a = 1.0 / shape_a.mass[collision][approaching]
                        inv_mass_b = 1.0 / shape_b.mass[collision][approaching]
                        j /= inv_mass_a + inv_mass_b

                        # Apply impulse to velocities
                        shape_a.velocity[update_indices] -= normal[approaching] * j.view(-1, 1) * inv_mass_a.view(-1, 1)
                        shape_b.velocity[update_indices] += normal[approaching] * j.view(-1, 1) * inv_mass_b.view(-1, 1)

    def to(self, device: torch.device) -> 'PhysicsEngine':
        """Move physics engine to specified device."""
        return PhysicsEngine(
            self.environment.to(device),
            self.dt,
            self.substeps,
            self.collisions_enabled
        )

