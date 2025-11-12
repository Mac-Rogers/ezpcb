import numpy as np
import math
import pygame as pg

g = 0
dt = 0.005 #s
pxPerMeter = 20

screen_width = 720
screen_height = 480


def collisionHandler(node, walls):
    point = node.getPos()
    velocity = np.array(node.getVel())

    for wall in walls:
        # Wall endpoints
        A = np.array([wall.start_x, wall.start_y])
        B = np.array([wall.end_x, wall.end_y])
        AB = B - A
        AB_length = np.linalg.norm(AB)
        if AB_length == 0:
            continue

        # Unit vector along wall
        u = AB / AB_length

        # Vector from A to point
        AP = np.array(point) - A

        # Project AP onto wall
        t = np.dot(AP, u)
        t_clamped = np.clip(t, 0, AB_length)
        closest = A + u * t_clamped  # closest point on wall segment

        # Normal vector
        n = np.array([-u[1], u[0]])  # 90 deg CCW
        n = n / np.linalg.norm(n)

        # Distance from point to wall
        delta = np.array(point) - closest
        dist = np.dot(delta, n)

        # Collision threshold
        threshold = 0.5
        if dist < threshold:
            # --- 1. Positional Correction ---
            correction = (threshold - dist) * n
            node.x += correction[0]
            node.y += correction[1]

            # --- 2. Velocity Correction ---
            v = velocity
            vn = np.dot(v, n)  # component into wall

            if vn < 0:  # only if moving into wall
                vt = v - vn * n  # tangential velocity remains
                vr = -vn * wall.coef_rest * n  # reflected normal component
                node.updateVel(*(vt + vr))
            else:
                node.updateVel(*v)

            return True  # collision handled

    return False

class Node:
    def __init__(self, pos, vel, mass, constraint="free", collisions=True):
        '''
        pos = (x, y)
        vel = (v_x, v_y)
        mass = m
        constraint:
        - "fixed" -> cannot move
        - "free"  -> can be moved by another object in any way
        - "vert"  -> can only be moved vertically
        - "hori"  -> can only be moved horizontally
        collisions = True -> enables collision handler with walls
        '''
        self.object = object
        self.x = pos[0]
        self.y = pos[1]
        self.v_x = vel[0]
        self.v_y = vel[1]
        self.a_x = 0
        self.a_y = 0
        self.mass = mass
        self.constraint = constraint
        self.collisions = collisions
        self.objects = []

        nodes.append(self)

    def update(self):
        self.step()

        if self.collisions:
            collisionHandler(self, walls)

        pg.draw.circle(screen, "red", (self.x * pxPerMeter, screen_height - self.y * pxPerMeter), 5)
    
    def getPos(self):
        return self.x, self.y
    
    def getVel(self):
        return self.v_x, self.v_y

    def getTotalForce(self):
        # gravity from mass which is a property of the node
        F_x = 0
        F_y = self.mass * g

        # add other forces
        for object in self.objects:
            F_x += object.getForce(self)[0]
            F_y += object.getForce(self)[1]
        #print(F_x, F_y)
        return F_x, F_y
    
    def updateVel(self, v_x, v_y):
        self.v_x = v_x
        self.v_y = v_y

    def step(self):
        F_x, F_y = self.getTotalForce()

        self.a_x = F_x / self.mass
        self.a_y = F_y / self.mass
        
        self.v_x += self.a_x * dt
        self.v_y += self.a_y * dt

        if self.constraint == "fixed":
            self.v_x = 0
            self.v_y = 0
        if self.constraint == "vert":
            self.v_x = 0
        if self.constraint == "hori":
            self.v_y = 0

        self.x += self.v_x * dt
        self.y += self.v_y * dt

        return self.x, self.y
    
    def attachObject(self, object):
        self.objects.append(object)

class Wall:
    def __init__(self, start_pos, end_pos, coef_restitution=1):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.end_x = end_pos[0]
        self.end_y = end_pos[1]
        self.coef_rest = coef_restitution

        walls.append(self)
    
    def render(self):
        pg.draw.line(screen, "blue", (self.start_x * pxPerMeter, screen_height - self.start_y * pxPerMeter), (self.end_x * pxPerMeter, screen_height - self.end_y * pxPerMeter), 5)

class Spring:
    def __init__(self, node1, node2, k, rest_length=0):
        self.node1 = node1
        self.node2 = node2
        self.k = k
        self.rest_length = rest_length
        self.v_x = 0
        self.v_y = 0
        self.a_y = 0
        self.y = 5

        node1.attachObject(self)
        node2.attachObject(self)

        springs.append(self)
    
    def render(self):
        pg.draw.line(screen, "yellow", (self.node1.getPos()[0] * pxPerMeter - 3, screen_height - self.node1.getPos()[1] * pxPerMeter), (self.node2.getPos()[0] * pxPerMeter - 3, screen_height - self.node2.getPos()[1] * pxPerMeter))
    
    def getForce(self, node):
        pos1 = np.array(self.node1.getPos())
        pos2 = np.array(self.node2.getPos())

        r = pos2 - pos1
        length = np.linalg.norm(r)
        if length == 0:
            return (0, 0)

        # spring direction
        direction = r / length

        # assume zero rest length
        force_magnitude = self.k * (length - self.rest_length)
        force_vector = force_magnitude * direction

        # apply force to the right node
        if node == self.node1:
            return force_vector  # Force on node1 due to node2
        elif node == self.node2:
            return -force_vector  # Newton’s 3rd law
        else:
            return (0, 0)
        
class Damper:
    def __init__(self, node1, node2, c):
        self.node1 = node1
        self.node2 = node2
        self.c = c
        self.v_x = 0 
        self.v_y = 0
        self.a_y = 0
        self.y = 5

        node1.attachObject(self)
        node2.attachObject(self)

        dampers.append(self)
    
    def render(self):
        pg.draw.line(screen, "green", (self.node1.getPos()[0] * pxPerMeter + 3, screen_height - self.node1.getPos()[1] * pxPerMeter), (self.node2.getPos()[0] * pxPerMeter + 3, screen_height - self.node2.getPos()[1] * pxPerMeter))
    
    def getForce(self, node):
        vel1 = np.array(self.node1.getVel())
        vel2 = np.array(self.node2.getVel())

        v = vel2 - vel1
        speed = np.linalg.norm(v)
        if speed == 0:
            return (0, 0)

        # spring direction
        direction = v / speed

        # assume zero rest length
        force_magnitude = self.c * speed
        force_vector = force_magnitude * direction

        # apply force to the right node
        if node == self.node1:
            return force_vector  # Force on node1 due to node2
        elif node == self.node2:
            return -force_vector  # Newton’s 3rd law
        else:
            return (0, 0)
    
class MouseNode(Node):
    def step(self):
        # Override to disable physics stepping for this node
        return self.x, self.y

nodes = []
springs = []
dampers = []
walls = []
members = []




dragging = False
mouse_x, mouse_y = 0, 0

while running:
    # poll for events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_x, mouse_y = event.pos
                for node in nodes:
                    node_pos = node.getPos()
                    dist = math.hypot(node_pos[0] * pxPerMeter - mouse_x, screen_height - node_pos[1] * pxPerMeter - mouse_y)
                    if dist < 10:  # If the click is within 10 pixels of the node
                        dragging = True
                        #dragged_node = node
                        # attach a spring between the mouse and the node
                        mouse_node = MouseNode((mouse_x/pxPerMeter, (screen_height - mouse_y)/pxPerMeter), (0,0), 0.001, "free", False)
                        mouse_spring = Spring(mouse_node, node, 50, 0)
                        mouse_damper = Damper(mouse_node, node, 5)
                        break
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                dragging = False
                #dragged_node = None
                if 'mouse_spring' in locals():
                    springs.remove(mouse_spring)
                    mouse_spring.node1.objects.remove(mouse_spring)
                    mouse_spring.node2.objects.remove(mouse_spring)
                    del mouse_spring
                    nodes.remove(mouse_node)
                    del mouse_node
                if 'mouse_damper' in locals():
                    dampers.remove(mouse_damper)
                    del mouse_damper
        
        elif event.type == pg.MOUSEMOTION:
            if dragging and 'mouse_spring' in locals():
                mouse_x, mouse_y = event.pos
                new_x = mouse_x / pxPerMeter
                new_y = (screen_height - mouse_y) / pxPerMeter
                mouse_spring.node1.x = new_x
                mouse_spring.node1.y = new_y


    screen.fill("black")


    for node in nodes:
        node.update()
    
    for spring in springs:
        spring.render()
    
    for damper in dampers:
        damper.render()

    for wall in walls:
        wall.render()


    pg.display.flip()
    clock.tick(1/dt) # 100fps is the max

pg.quit()
