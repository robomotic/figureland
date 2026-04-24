
def get_force_vector(self, position, dt):
    force = torch.zeros_like(position)
    force[:, 1] -= self.gravity
    force -= self.air_resistance * position
    return force

import types
from figureland.physics import Environment
Environment.get_force_vector = get_force_vector

from figureland.shapes import Square
from figureland.physics import Environment, PhysicsEngine
from figureland import SimulationExporter

RESOLUTION = (100, 400)
FRAMES = 200
FPS = 30

env = Environment(bounds=(-1.0, 1.0), gravity=9.8)
engine = PhysicsEngine(env)
exporter = SimulationExporter(RESOLUTION, fps=FPS)

shapes = [
    Square.from_random(1, (-1,1), (0.1,0.1), (1,1), (0.8,0.8), seed=42),
    Square.from_random(1, (-1,1), (0.1,0.1), (1,1), (0.8,0.8), seed=43)
]

shapes[0].position[0] = [-0.5, 0.8]
shapes[1].position[0] = [0.5, 0.8]
shapes[0].color[:] = [1.0, 0.0, 0.0]
shapes[1].color[:] = [0.0, 0.0, 1.0]

env.add_shapes(shapes)
exporter.save_frame(env.shapes, 'first_frame.png')

for _ in range(FRAMES):
    exporter.add_frame(env.shapes)
    engine.step()

exporter.save_frame(env.shapes, 'last_frame.png')
exporter.save_video('falling_balls.mp4')
print('✅ All output generated successfully in ./output/')

