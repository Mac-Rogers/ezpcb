import numpy as np
import pygame as pg
import math
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import to_pygame
from pymunk import Vec2d, SpaceDebugDrawOptions as SDO
from pymunk.space_debug_draw_options import SpaceDebugColor


'''
TODO:
- fix spring connections based on shortest distance
- update the spring connections when components move
- display pads
'''

'''
a design has many components
a design has many nets
a component has many pads
a net has many pads
a net has many vias
a net has many wires

via has (vias are added independently of the DSN, ignore vias in dsn)
    - 1 position
    - 1 id (unique)

pad has 
    - 1 position
    - 1 component
    - 1 net
    - 1 shape
    - 1 name (not unique)
    - 1 id (unique)
    - many layers (such as a through-hole pad (not connected as a via))

component has
    - 1 id (not the same as component name in DSN)
    - many pads

net has
    - 1 name (unique)
    - many pads
    - many vias

wire has
    - many segments
    - 1 net

    
if 2 pads have the same shape, they are defined in the "library" under the same name.
The shape is then defined under the non-unique name as a padstack
'''

pads = []
vias = []
nets = []
components = []

boundary = [] # list of [x,y] points defining the boundary polygon

########################## PARAMETERS ##########################
BOARD_WIDTH = 5

class Pad:
    def __init__(self, id, name, position, shape_type, shape, component, net, layers):
        self.ID = int(id)
        self.name = name
        self.position = position # [x,y]
        self.shape_type = shape_type # "circle" or "polygon"
        self.shape = shape
        self.component = component  # component object
        self.net = net  # net object
        self.layers = layers # [1, 2] 

    def getID(self):
        return self.ID
    
    def getPosition(self):
        return self.position

    def getShape(self):
        return self.shape
    
    def getComponent(self):
        return self.component
    
    def getNet(self):
        return self.net
    
    def getLayers(self):
        return self.layers
    
    def updatePosition(self, newPosition):
        self.position = newPosition
    
    def updateLayers(self, newLayers):
        self.layers = newLayers
    
    def setShape(self, newShape):
        # only used initially when reading in the DSN file
        self.shape = newShape
    
    def setShapeType(self, newShapeType):
        # only used initially when reading in the DSN file
        self.shape_type = newShapeType

    def setNet(self, newNet):
        # only used initially when reading in the DSN file
        self.net = newNet
    
    def setComponent(self, newComponent):
        # only used initially when reading in the DSN file
        self.component = newComponent
    
    def setLayers(self, newLayer):
        # only used initially when reading in the DSN file
        self.layers.append(newLayer)

class Via:
    def __init__(self, id, position):
        self.ID = int(id)
        self.position = position

    def getID(self):
        return self.ID

    def getPosition(self):
        return self.position
    
    def updatePosition(self, newPosition):
        self.position = newPosition

class Net:
    def __init__(self, name, net_pads, net_vias):
        self.name = name
        self.pads = net_pads  # list of pad objects
        self.vias = net_vias  # list of via objects
    
    def getName(self):
        return self.name
    
    def getPads(self):
        return self.pads
    
    def getVias(self):
        return self.vias
    
    def addVia(self, via):
        self.vias.append(via)
    
    def getPoints(self):
        points = []
        for pad in self.pads:
            points.append(pad.getPosition())
        for via in self.vias:
            points.append(via.getPosition())
        return points

    def addSprings(self):
        # each component is a node, connect each node to every other node in the net with a spring.
        
        for pad in self.pads:
            compID = pad.getComponent()
            comp = next((c for c in components if c.getID() == compID), None)
            print(comp)
            # Connect this component to every other component in the net
            for other_pad in self.pads:
                if other_pad != pad:
                    other_compID = other_pad.getComponent()
                    other_comp = next((c for c in components if c.getID() == other_compID), None)

                    #if other_comp:
                    #    Spring(comp.node, other_comp.node, 10)
                    #    Damper(comp.node, other_comp.node, 1)

class Component:
    def __init__(self, id, comp_pads):
        self.id = id
        self.pads = comp_pads  # list of pad objects
        self.node = None  # physics node attached to the component

    def getID(self):
        return self.id

    def getPads(self):
        return self.pads

    def getBoundingBox(self):
        if not self.pads:
            return None  # No pads in the component
        # bounding box includes the shape of the pads
        for pad in self.pads:
            if pad.shape_type == "circle":
                diameter = pad.shape[0][1]
                pad_min_x = pad.getPosition()[0] - diameter / 2
                pad_max_x = pad.getPosition()[0] + diameter / 2
                pad_min_y = pad.getPosition()[1] - diameter / 2
                pad_max_y = pad.getPosition()[1] + diameter / 2
            elif pad.shape_type == "polygon":
                xs = [vertex[0] + pad.getPosition()[0] for vertex in pad.shape]
                ys = [vertex[1] + pad.getPosition()[1] for vertex in pad.shape]
                pad_min_x = min(xs)
                pad_max_x = max(xs)
                pad_min_y = min(ys)
                pad_max_y = max(ys)
            else:
                continue  # Unknown shape type, skip

            if 'min_x' not in locals():
                min_x, max_x = pad_min_x, pad_max_x
                min_y, max_y = pad_min_y, pad_max_y
            else:
                min_x = min(min_x, pad_min_x)
                max_x = max(max_x, pad_max_x)
                min_y = min(min_y, pad_min_y)
                max_y = max(max_y, pad_max_y)
        
        min_x -= BOARD_WIDTH / 2
        max_x += BOARD_WIDTH / 2
        min_y -= BOARD_WIDTH / 2
        max_y += BOARD_WIDTH / 2

        return (min_x, -max_y), (max_x, -min_y)

    def move(self, x, y):
        for pad in self.pads:
            pos = pad.getPosition()
            newPos = [pos[0] + x, pos[1] + y]
            pad.updatePosition(newPos)
    
    def getPos(self):
        # position is the center of the bounding box
        p1, p2 = self.getBoundingBox()
        #print(p1, p2)
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        return (center_x, center_y)

    def attachNode(self):
        # attach a node to the center of the component
        (x,y) = self.getPos()
        #print(x,y)
        
        #self.node = Node((x / pxPerMeter * 3, (screen_height - y * 3) / pxPerMeter), (0,0), 1, "free", True)


class Wire:
    def __init__(self, net, segments):
        self.net = net  # net object
        self.segments = segments  # list of segments, each segment is [[x1,y1],[x2,y2]]
    
    def getNet(self):
        return self.net
    
    def getSegments(self):
        return self.segments
    
    def addSegment(self, segment):
        self.segments.append(segment)


def parse_shape(shape_line):
    '''
    helper function of processDSNfile to parse a shape line
    example of shape_line:
       (shape(circle 1 6.5 0 0))
       (shape(polygon 1 0.01 -3.9371 1.9685 3.9369 1.9685 3.9369 -1.9685 -3.9371 -1.9685))
    
    (polygon
        <layer_id>
        <aperture_width>
        {<vertex>}
        [(aperture_type [round | square])]
    )
    '''
    shape_type = ""
    shape = []
    layer = 0

    words = shape_line.split()
    # remove the "))" at the end of the last word
    words[-1] = words[-1].replace("))", "")
    #print(words)

    layer = int(words[1])

    if "circle" in shape_line:
        shape_type = "circle"

        # get the diameter
        #print(shape_line)
        diameter = float(words[2])
        shape.append(["diameter", diameter])
        #print(f"Parsed circle shape: {shape}")

    elif "polygon" in shape_line:
        shape_type = "polygon"

        # ignore the aperture width
        # get pairs of vertices
        for i in range(3, len(words), 2):
            x = float(words[i])
            y = float(words[i+1])
            shape.append([x, y])
        #print(f"Parsed polygon shape: {shape}")


    return shape_type, shape, layer


def processDSNfile(file_name):
    '''
    Top level design:

    (pcb <pcb_id>
        [<parser_descriptor>]
        [<capacitance_resolution_descriptor>]
        [<conductance_resolution_descriptor>]
        [<current_resolution_descriptor>]
        [<inductance_resolution_descriptor>]
        [<resistance_resolution_descriptor>]
        [<resolution_descriptor>]
        [<time_resolution_descriptor>]
        [<voltage_resolution_descriptor>]
        [<unit_descriptor>]
        [<structure_descriptor> | <file_descriptor>]
        [<placement_descriptor> | <file_descriptor>]
        [<library_descriptor> | <file_descriptor>]
        [<floor_plan_descriptor> | <file_descriptor>]
        [<part_library_descriptor> | <file_descriptor>]
        [<network_descriptor> | <file_descriptor>]
        [<wiring_descriptor>]
        [<color_descriptor>]
    )

    Only care for structure, library and network. Second level design:

    (structure
        [<unit_descriptor> | <resolution_descriptor> | null]
        {<layer_descriptor>}
        [<layer_noise_weight_descriptor>]
        {<boundary_descriptor>}
        {<place_boundary_descriptor>}
        [{<plane_descriptor>}]
        [{<region_descriptor>}]
        [{<keepout_descriptor>}]
        <via_descriptor>
        [<control_descriptor>]
        <rule_descriptor>
        [<structure_place_rule_descriptor>]
        {<grid_descriptor>}
    )

    (library
        [<unit_descriptor>]
        {<image_descriptor>}
        [{<jumper_descriptor>}]
        {<padstack_descriptor>}
        [<directory_descriptor>]
        [<extra_image_directory_descriptor>]
        [{<family_family_descriptor>}]
        [{<image_image_descriptor>}]
    )

    (network
        {<net_descriptor>}
        [{<class_descriptor>}]
        [{<class_class_descriptor>}]
        [{<group_descriptor>}]
        [{<group_set_descriptor>}]
        [{<pair_descriptor>}]
        [{<bundle_descriptor>}]
    )
    '''
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # remove the \n from each line
    lines = [line.strip() for line in lines]

    ######################## STRUCTURE ########################

    # find the number of layers
    num_layers = 0
    for line in lines:
        if line.startswith("(layer"):
            num_layers += 1

        if line.startswith("(boundary"):
            #    (boundary(path signal 0 0.2927 -141.7319 -0.0001 -141.439 -0.0001 -0.2927 0.2927 0.0001 208.368 0.0001 208.6609 -0.2927 208.6609 -141.439 208.368 -141.7319 0.2927 -141.7319 )

            words = line.split()
            # remove the last word from the line
            words.pop()
            
            for i in range(3, len(words), 2):
                x = float(words[i])
                y = float(words[i+1])
                boundary.append([x, y])
            print(f"Parsed boundary: {boundary}")
    # don't get confused with pins in network so stop before then
    network_line = 0
    for i in range(len(lines)):
        if lines[i].startswith("(network"):
            network_line = i
            break

    # create a pad using the line "(pin p39 39 28.2678 -80.4722)"
    for line in lines[:network_line]: # stop before network
        if line.startswith("(pin"):
            # split the line into words
            words = line.split()
            # remove the ")" at the end of the last word
            words[-1] = words[-1].replace(")", "")

            pad_name = words[1]
            pad_id = words[2]
            pad_x = float(words[3])
            pad_y = float(words[4])
            pad_position = [pad_x, pad_y]

            pad = Pad(pad_id, pad_name, pad_position, None, None, None, None, [])
            pads.append(pad)
    
    # we dgaf about via shape

    # padstacks: 
    #   (at this point pads have been created)
    #   (padstack p39
    #     (shape(circle 1 21.6535 0 0))
    #     (shape(circle 2 21.6535 0 0))
    #   )
    for i in range(len(lines)):
        line = lines[i]
        
        if line.startswith("(padstack"):
            # split the line into words
            words = line.split()
            padstack_name = words[1]

            # the next lines will be the shapes
            i += 1
            while i < len(lines) and lines[i].startswith("(shape"):
                shape_line = lines[i]
                # parse the shape line
                shape_type, shape, layer = parse_shape(shape_line)
                #print(words, layer)
                # find the corresponding pad and set its shape
                for pad in pads:
                    if pad.name == padstack_name:
                        pad.setShape(shape)
                        pad.setShapeType(shape_type)
                        pad.setLayers(layer)
                i += 1

    ######################### NETWORK #########################
    for i in range(network_line + 1, len(lines)): # start from network
        line = lines[i]
        if line.startswith("(net"):
            '''
            (net GND
                (pins u1-54 u1-474 u1-555 u1-933)
            )
            '''
            # split the line into words
            words = line.split()
            # remove the ")" at the end of the last word
            words[-1] = words[-1].replace(")", "")
            net_name = words[1]

            # get the next line
            next_line = lines[i+1]
            # split the next line into words
            next_words = next_line.split()
            # remove the ")" at the end of the last word
            next_words[-1] = next_words[-1].replace(")", "")

            net_pads = []  # list of pad objects

            # get the pins from the next line
            pins = next_words[1:]
            for pin in pins:
                # remove the "u1-" from the pin name by splitting at the "-"
                pin = pin.split("-")[1]
                
                # find the corresponding pad using the pin id
                for pad in pads:
                    if pad.ID == int(pin):
                        net_pads.append(pad)
                        # if found, set the net of the pad
                        pad.setNet(net_name)
                        #print(f"Set net of pad {pad.ID} to {words[1]}")

            # create a net object
            net = Net(net_name, net_pads, [])
            nets.append(net)
    
    ######################### COMPONENTS #########################
    # if the difference between two consecutive pad ids is < 16, they belong to the same component
    pads_sorted = sorted(pads, key=lambda pad: pad.ID)
    current_component_id = 0
    current_component_pads = []

    for i in range(len(pads_sorted)):
        pad = pads_sorted[i]
        if i == 0:
            current_component_pads.append(pad)
        else:
            prev_pad = pads_sorted[i-1]
            if pad.ID - prev_pad.ID < 16:
                current_component_pads.append(pad)
            else:
                # create a component object
                component = Component(current_component_id, current_component_pads)
                components.append(component)
                # set the component of each pad
                for p in current_component_pads:
                    p.setComponent(current_component_id)
                # reset for the next component
                current_component_id += 1
                current_component_pads = [pad]
    
    # Handle the last component (if there are any pads remaining)
    if current_component_pads:
        component = Component(current_component_id, current_component_pads)
        components.append(component)
        # set the component of each pad
        for p in current_component_pads:
            p.setComponent(current_component_id)
        
        
def printStructure():
    print("Components:")
    for component in components:
        print(f"  Component {component.id}:")
        for pad in component.pads:
            print(f"    Pad {pad.getID()}: {pad.getPosition()}, on layers {pad.getLayers()}, in net {pad.getNet()}")
    print("\nNets:")
    for net in nets:
        print(f"  Net {net.getName()}:")
        for pad in net.getPads():
            print(f"    Pad {pad.getID()}: {pad.getPosition()}, on layers {pad.getLayers()}, in net {pad.getNet()}")



class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, surface):
        super().__init__(surface)
        self.color_outline        = (255, 255, 255)
        self.color_outline_static = (150, 150, 150)

    def color_for_shape(self, shape):
        """Return fill color based on the shape/body mass."""
        mass = getattr(shape.body, "mass", 0.0)

        if abs(mass - 1.0) < 1e-6:
            # “normal” boxes
            return SpaceDebugColor(255, 255, 255, 255)  # cyan-ish
        else:
            return SpaceDebugColor(150, 150, 150, 255)  # gray

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [pymunk.pygame_util.to_pygame(v, self.surface) for v in verts]

        # use the fill_color chosen by color_for_shape()
        pg.draw.polygon(self.surface, fill_color.as_int(), ps, 3)

        # outline color you want
        #pg.draw.polygon(self.surface, self.color_outline, ps, 3)


def drawComponents():
    for component in components:
        for pad in component.getPads():
            pos = pad.getPosition()
            x = int(pos[0])
            y = int(-pos[1])
            if pad.getLayers() == [1]:  # only draw pads on layer 1
                color = "red"
            elif pad.getLayers() == [2]:  # only draw pads on layer 2
                color = "blue"
            elif 1 in pad.getLayers() and 2 in pad.getLayers():  # draw pads on both layers
                color = "purple"
            else:
                color = "green"
            
            if pad.shape_type == "circle":
                diameter = pad.shape[0][1]
                pg.draw.circle(screen, color, (int(x * zoom + dx), int(y * zoom + dy)), int(diameter/2 * zoom), 0)
            elif pad.shape_type == "polygon":
                points = []
                for vertex in pad.shape:
                    vx = int((vertex[0] + pad.getPosition()[0]) * zoom + dx)
                    vy = int((-vertex[1] - pad.getPosition()[1]) * zoom + dy)
                    points.append((vx, vy))
                pg.draw.polygon(screen, color, points, 0)
        
        p1, p2 = component.getBoundingBox()
        #print(p1, p2)
        if p1 is not None and p2 is not None:
            x1 = int(p1[0] * zoom + dx)
            y1 = int(p1[1] * zoom + dy)
            x2 = int(p2[0] * zoom + dx)
            y2 = int(p2[1] * zoom + dy)
            pg.draw.rect(screen, "yellow", (x1, y1, x2 - x1, y2 - y1), 1)

    # draw the boundary
    if len(boundary) > 1:
        boundary_points = []
        for point in boundary:
            bx = int(zoom * point[0] + dx)
            by = int(zoom * -point[1] + dy)
            boundary_points.append((bx, by))
        pg.draw.polygon(screen, "grey", boundary_points, 1)

def drawNets():
    for net in nets:
        points = net.getPoints()
        
        # for each point, find the next closest point and draw a line to it
        for i in range(len(points)):
            p1 = points[i]
            min_dist = float('inf')
            closest_point = None
            for j in range(len(points)):
                if i == j:
                    continue
                p2 = points[j]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = p2
            if closest_point is not None:
                x1 = int(p1[0] * zoom + dx)
                y1 = int(-p1[1] * zoom + dy)
                x2 = int(closest_point[0] * zoom + dx)
                y2 = int(-closest_point[1] * zoom + dy)
                pg.draw.line(screen, "white", (x1, y1), (x2, y2), 1)

def getBodyAt(x, y, bodies, shapes):
    for i in range(len(bodies)):
        width = -(shapes[i].get_vertices()[2].x - shapes[i].get_vertices()[0].x)
        height = shapes[i].get_vertices()[1].y - shapes[i].get_vertices()[0].y
        size = min(width, height)

        if bodies[i].position.x - size/2 <= x <= bodies[i].position.x + size/2 and \
           bodies[i].position.y - size/2 <= y <= bodies[i].position.y + size/2:
            return i, bodies[i]
    return None, None



def addPhysicsObjects(space):
    bodies = []
    shapes = []
    traces = []
    component_to_body = {}  # map component id to physics body
    component_centers = {}

    # create one physics body per component and remember its center
    for component in components:
        p1, p2 = component.getBoundingBox()
        width = p2[0] - p1[0]
        height = p2[1] - p1[1]
        mass = 1
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        shape = pymunk.Poly.create_box(body, (width, height))
        space.add(body, shape)

        x, y = component.getPos()
        body.position = Vec2d(x, y)
        body.angle = 0

        bodies.append(body)
        shapes.append(shape)
        component_to_body[component.getID()] = body
        component_centers[component.getID()] = Vec2d(x, y)

    # create springs between pads in the same net
    for net in nets:
        points = net.getPoints() # points are [x,y] positions of pads and vias
        print(points)

        # find the closest pair of points and connect them with a spring
        # then remove the first point and repeat until all points are connected
        for point in points:
            dist = float('inf')
            closest_point = None
            for other_point in points:
                if point == other_point:
                    continue
                d = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if d < dist:
                    dist = d
                    closest_point = other_point
            if closest_point is not None:
                # find the components of the two points
                pad1 = next((pad for pad in pads if pad.getPosition() == point), None)
                pad2 = next((pad for pad in pads if pad.getPosition() == closest_point), None)
                if pad1 is None or pad2 is None:
                    continue
                comp_id1 = pad1.getComponent()
                comp_id2 = pad2.getComponent()
                if comp_id1 == comp_id2:
                    continue

                body1 = component_to_body.get(comp_id1)
                body2 = component_to_body.get(comp_id2)
                if body1 is None or body2 is None:
                    continue

                # Convert pad positions into the physics coordinate system
                # used by bodies (pad Y is negated in drawing/physics elsewhere).
                pad1_world = Vec2d(pad1.getPosition()[0], -pad1.getPosition()[1])
                pad2_world = Vec2d(pad2.getPosition()[0], -pad2.getPosition()[1])

                # anchors are local offsets from the body's position
                anchor1 = pad1_world - component_centers[comp_id1]
                anchor2 = pad2_world - component_centers[comp_id2]

                rest_length = 0#(pad1_world - pad2_world).length
                stiffness = 10.0
                damping = 5.0

                spring = pymunk.DampedSpring(body1, body2, anchor1, anchor2, rest_length, stiffness, damping)
                space.add(spring)
                traces.append(spring)




    return bodies, shapes, traces, component_to_body



if __name__ == "__main__":
    processDSNfile("DSN/mosfetDriver.dsn")

    #printStructure()

    pg.init()
    fps = 60
    WIDTH, HEIGHT = 800, 600
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    running = True

    for component in components:
        component.attachNode()
    
    for net in nets:
        net.addSprings()


    space = pymunk.Space()
    space.gravity = 0, 0
    space.sleep_time_threshold = 0.3
    space.damping = 0.1

    #draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options = CustomDrawOptions(screen)
    draw_options.flags &= ~SDO.DRAW_CONSTRAINTS
    pymunk.pygame_util.positive_y_is_up = False

    total_time = 0
    zoom = 1.0
    is_dragging = False
    is_component_dragging = False
    offset_x, offset_y = 0, 0
    offset_x_component, offset_y_component = 0, 0
    selected_body = None
    disable_body_while_dragging = None
    dx, dy = 0, 0
    dx_component, dy_component = 0, 0
    mouse_x, mouse_y = 0, 0
    last_click_time = 0
    double_click_threshold = 300  # milliseconds
    static_bodies = []

    bodies, shapes, traces, component_to_body = addPhysicsObjects(space)

    # mouse drag state for left-click dragging
    grabbed_body = None
    grab_joint = None
    grabbed_was_static = False

    # kinematic body that follows the mouse for dragging
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    space.add(mouse_body)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right mouse button (pan)
                    is_dragging = True
                    offset_x = event.pos[0] - dx
                    offset_y = event.pos[1] - dy

                elif event.button == 1:  # Left mouse button (select/drag or double-click)
                    current_time = pg.time.get_ticks()
                    mouse_x, mouse_y = event.pos
                    # convert to world coords used by bodies
                    world_x = (mouse_x - dx) / zoom
                    world_y = (mouse_y - dy) / zoom

                    if current_time - last_click_time < double_click_threshold:
                        # Double click detected -> toggle fixed state
                        body_index, body = getBodyAt(world_x, world_y, bodies, shapes)
                        if body is not None and body not in static_bodies:
                            #toggleFixed(body)
                            static_bodies.append(body)
                            body.mass = 1000000
                            body.moment = 1000000
                        elif body is not None and body in static_bodies:
                            static_bodies.remove(body)
                            body.mass = 1
                            body.moment = pymunk.moment_for_box(body.mass, (10, 10))
                    else:
                        # start dragging (single click)
                        body_index, body = getBodyAt(world_x, world_y, bodies, shapes)
                        if body is not None:
                            grabbed_body = body

                            # if body is static, make it dynamic temporarily for dragging
                            if body in static_bodies:
                                body.mass = 1
                                disable_body_while_dragging = body

                            # position the kinematic mouse body and attach a pivot joint
                            mouse_body.position = Vec2d(world_x, world_y)
                            # anchor on grabbed body in local coordinates
                            try:
                                local_anchor = grabbed_body.world_to_local((world_x, world_y))
                            except Exception:
                                local_anchor = (0, 0)
                            grab_joint = pymunk.PivotJoint(mouse_body, grabbed_body, (0, 0), local_anchor)
                            grab_joint.max_force = 500000
                            space.add(grab_joint)

                    last_click_time = current_time

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 3:
                    is_dragging = False
                elif event.button == 1:
                    # release any grabbed body
                    if grab_joint is not None:
                        try:
                            space.remove(grab_joint)
                        except Exception:
                            pass
                        grab_joint = None
                    if grabbed_body in static_bodies:
                        # restore static state
                        disable_body_while_dragging = None
                    grabbed_body = None

            elif event.type == pg.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                # update kinematic mouse body position in world coords
                world_x = (mouse_x - dx) / zoom
                world_y = (mouse_y - dy) / zoom
                try:
                    mouse_body.position = Vec2d(world_x, world_y)
                except Exception:
                    pass
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

        for body in static_bodies:
            if disable_body_while_dragging == body:
                continue

            body.mass = 1000000
            body.moment = 1000000
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0
            body.force = (0.0, 0.0)
            body.torque = 0.0

            angle = body.angle
            nearest_90 = round(angle / (math.pi / 2)) * (math.pi / 2)
            body.angle = nearest_90
                
        screen.fill("black")

        space.debug_draw(draw_options)

     
        for c in space.constraints:
            if isinstance(c, pymunk.DampedSpring):
                p11, p12 = c.a.local_to_world(c.anchor_a)
                p21, p22 = c.b.local_to_world(c.anchor_b)
                p11 = (p11) * zoom + dx
                p12 = (p12) * zoom + dy
                p21 = (p21) * zoom + dx
                p22 = (p22) * zoom + dy
                p1 = Vec2d(p11, p12)
                p2 = Vec2d(p21, p22)
                pg.draw.line(screen, (200, 0, 0), to_pygame(p1, screen), to_pygame(p2, screen), 2)
   

        pg.display.flip()
        dt = clock.tick(fps)
        total_time += dt / 1000.0