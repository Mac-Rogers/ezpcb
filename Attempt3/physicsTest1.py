import sys
import pygame as pg
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

fps = 60
pg.init()
WIDTH, HEIGHT = 800, 600
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()

clock.tick(1)

space = pymunk.Space()
space.gravity = 0, 0
space.sleep_time_threshold = 0.3

draw_options = pymunk.pygame_util.DrawOptions(screen)
pymunk.pygame_util.positive_y_is_up = False


def car(space):
    pos = Vec2d(100, 200)

    wheel_color = 52, 219, 119, 255
    mass = 100
    radius = 25
    moment = pymunk.moment_for_circle(mass, 20, radius)
    wheel1_b = pymunk.Body(mass, moment)
    wheel1_s = pymunk.Circle(wheel1_b, radius)
    wheel1_s.friction = 1.5
    wheel1_s.color = wheel_color
    space.add(wheel1_b, wheel1_s)

    mass = 100
    radius = 25
    moment = pymunk.moment_for_circle(mass, 20, radius)
    wheel2_b = pymunk.Body(mass, moment)
    wheel2_s = pymunk.Circle(wheel2_b, radius)
    wheel2_s.friction = 1.5
    wheel2_s.color = wheel_color
    space.add(wheel2_b, wheel2_s)

    mass = 100
    size = (50, 30)
    moment = pymunk.moment_for_box(mass, size)
    chassi_b = pymunk.Body(mass, moment)
    chassi_s = pymunk.Poly.create_box(chassi_b, size)
    space.add(chassi_b, chassi_s)

    wheel1_b.position = pos - (55, 0)
    wheel2_b.position = pos + (55, 0)
    chassi_b.position = pos + (0, -25)

    space.add(
        pymunk.DampedSpring(wheel1_b, chassi_b, (0, 0), (-20, -15), 70, 10000, 0.1),
        pymunk.DampedSpring(wheel1_b, chassi_b, (0, 0), (-25, 15), 50, 10000, 0.1),
        pymunk.DampedSpring(wheel2_b, chassi_b, (0, 0), (20, -15), 70, 10000, 0.1),
        pymunk.DampedSpring(wheel2_b, chassi_b, (0, 0), (25, 15), 50, 10000, 0.1),
    )

    speed = 4
    space.add(
        pymunk.SimpleMotor(wheel1_b, chassi_b, speed),
        pymunk.SimpleMotor(wheel2_b, chassi_b, speed),
    )

def boundingBox(space):
    '''
    make a bounding box
    '''
    mass = 1
    size = (20, 20)
    moment = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, moment)
    shape = pymunk.Poly.create_box(body, size)
    space.add(body, shape)

    body.position = Vec2d(300, 300)


car(space)
boundingBox(space)

total_time = 0
zoom = 1.0
is_dragging = False
offset_x, offset_y = 0, 0
dx, dy = 0, 0
mouse_x, mouse_y = 0, 0

while True:
    for event in pg.event.get():
        if (event.type == pg.QUIT or event.type == pg.KEYDOWN and (event.key in [pg.K_ESCAPE, pg.K_q])):
            sys.exit(0)
        
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 3:  # Right mouse button
                is_dragging = True
                offset_x = event.pos[0] - dx
                offset_y = event.pos[1] - dy
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 3:
                is_dragging = False
        elif event.type == pg.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            if is_dragging:
                dx = event.pos[0] - offset_x
                dy = event.pos[1] - offset_y    
        elif event.type == pg.MOUSEWHEEL:
            mouse_x, mouse_y = pg.mouse.get_pos()
            # Calculate world position before zoom
            world_x_before = (dx - mouse_x) / zoom
            world_y_before = (dy - mouse_y) / zoom
            
            # Apply zoom
            if event.y > 0:  # scroll up
                zoom *= 1.1
            elif event.y < 0:  # scroll down
                zoom /= 1.1
            
            # Calculate world position after zoom
            world_x_after = (dx - mouse_x) / zoom
            world_y_after = (dy - mouse_y) / zoom

            # Adjust offset to keep world position under mouse
            dx -= (world_x_after - world_x_before) * zoom
            dy -= (world_y_after - world_y_before) * zoom


    draw_options.transform = pymunk.Transform.scaling(zoom).translated(dx / zoom, dy / zoom)

    space.step(1.0 / fps)

    screen.fill(pg.Color("black"))

    space.debug_draw(draw_options)

    for b in space.bodies:
        p = pymunk.pygame_util.to_pygame(b.position, screen)

    pg.display.flip()

    dt = clock.tick(fps)
    total_time += dt / 1000.0