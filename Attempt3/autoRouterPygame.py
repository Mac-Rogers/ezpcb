import numpy as np
import pygame as pg
import math
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import to_pygame
from pymunk import Vec2d, SpaceDebugDrawOptions as SDO
from pymunk.space_debug_draw_options import SpaceDebugColor
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


'''
TODO:
- first instance spring not generating correctly
- generate SES
- 2nd layer through vias
- bottom layer component selective collisions
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

ROUTE_DEBUG = False

pads = []
vias = []
nets = []
components = []
wires = []

boundary = [] # list of [x,y] points defining the boundary polygon
layers = [] # contains 2D arrays of Pixel obj representing what occupies each pixel of each layer

########################## PARAMETERS ##########################
BOARD_WIDTH = 5 # offset around components when calculating bounding box

def rotate(x, y, theta):
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

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
        # Keep an immutable copy of the original shape so rotations are
        # applied relative to the original vertices (avoid cumulative rotation).
        if shape is None:
            self.original_shape = None
        else:
            # make a deep copy of the vertex list
            self.original_shape = [list(v) for v in shape]

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
    
    def updatePosition(self, newPosition, theta):
        self.position = newPosition
        # apply rotations
        if self.shape_type == "polygon":
            # rebuild the rotated shape from the original vertex positions
            new_shape = []
            for v in self.original_shape:
                x, y = rotate(v[0], v[1], -theta)
                new_shape.append([x, y])
            self.shape = new_shape
        
    
    def updateLayers(self, newLayers):
        self.layers = newLayers
    
    def setShape(self, newShape):
        # only used initially when reading in the DSN file
        self.shape = newShape
        if newShape is None:
            self.original_shape = None
        else:
            self.original_shape = [list(v) for v in newShape]
    
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
        self.pads_offsets = {}  # map pad id to offset from component center
        for pad in self.pads:
            pad_pos = Vec2d(pad.getPosition()[0], -pad.getPosition()[1])
            comp_pos = Vec2d(self.getPos()[0], self.getPos()[1])
            offset = pad_pos - comp_pos
            self.pads_offsets[pad.getID()] = Vec2d(offset.x, -offset.y)
        print(f"comp pos: {self.getPos()}, pads offsets: {self.pads_offsets}")

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

    def move(self, x, y, t):
        for pad in self.pads:
            # rotate the pad_offsets by angle theta, then apply the shift of x and y
            x_offset_rot = self.pads_offsets[pad.getID()].x * np.cos(-t) - self.pads_offsets[pad.getID()].y * np.sin(-t)
            y_offset_rot = self.pads_offsets[pad.getID()].x * np.sin(-t) + self.pads_offsets[pad.getID()].y * np.cos(-t)
            pad.updatePosition([x + x_offset_rot, y + y_offset_rot], t)
    
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
        self.segments = segments  # list of segments, each segment is [[x1,y1],[x2,y2], layer, width]
        # one wire is typically used per pad-pad connection
        # store these two pads for convenience
        self.pad_1 = None
        self.pad_2 = None
    
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
            # find first integer token after pad name (robust to optional tokens like "(rotate 180)")
            pad_id = None
            id_index = None
            for idx in range(2, len(words)):
                tok = words[idx].replace(")", "")
                try:
                    pad_id_candidate = int(tok)
                    pad_id = str(pad_id_candidate)
                    id_index = idx
                    break
                except ValueError:
                    continue
            # fallback if no integer found (keep existing behaviour)
            if pad_id is None:
                pad_id = words[2]
                id_index = 2

            # parse coordinates after the id token
            try:
                pad_x = float(words[id_index + 1].replace(")", ""))
            except Exception:
                pad_x = 0.0
            try:
                pad_y = float(words[id_index + 2].replace(")", ""))
            except Exception:
                pad_y = 0.0

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


def constructSESfile(file_name):
    # oh dear
    pass


class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, surface):
        super().__init__(surface)
        self.color_outline        = (255, 255, 255)
        self.color_outline_static = (100, 100, 100)

    def color_for_shape(self, shape):
        """Return fill color based on the shape/body mass."""
        mass = getattr(shape.body, "mass", 0.0)

        if abs(mass - 1.0) < 1e-6:
            # “normal” boxes
            return SpaceDebugColor(255, 255, 0, 255)  # cyan-ish
        else:
            return SpaceDebugColor(100, 100, 100, 255)  # gray

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [pymunk.pygame_util.to_pygame(v, self.surface) for v in verts]

        # use the fill_color chosen by color_for_shape()
        pg.draw.polygon(self.surface, fill_color.as_int(), ps, 1)


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


def drawPygameComponents():
    # draw the boundary
    if len(boundary) > 1:
        boundary_points = []
        for point in boundary:
            bx = int(zoom * point[0] + dx)
            by = int(zoom * -point[1] + dy)
            boundary_points.append((bx, by))
        pg.draw.polygon(screen, "blue", boundary_points, 1)
    
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

def drawTraces():
    for wire in wires:
        for segment in wire.getSegments():
            # segment looks like ((x, y), (x, y), layer, width)
            if segment[2] == 1:
                colour = "red"
            elif segment[2] == 2:
                colour = "blue"
            else:
                colour = "green"
            start_pos = (int(zoom * segment[0][0] + dx), int(zoom * -segment[0][1] + dy))
            end_pos = (int(zoom * segment[1][0] + dx), int(zoom * -segment[1][1] + dy))
            width = int(segment[3] * zoom * 1)
            pg.draw.line(screen, colour, start_pos, end_pos, width)
        #print(wire.getSegments())

def addPhysicsObjects(space):
    bodies = []
    shapes = []
    traces = []
    component_to_body = {}  # map component id to physics body
    pad_to_spring = {}  # map pad id to spring
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
        #print(points)

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
                pad_to_spring[pad1.getID()] = spring
                pad_to_spring[pad2.getID()] = spring

    return bodies, shapes, traces, component_to_body, pad_to_spring


class Pixel:
    def __init__(self, x, y):
        '''
        x, y: top left corner of the pixel
        occupancy: None, "pad", "trace", "via", "air"
        layer: 1 (top), 2 (bottom), 3 (below 1), 4 (above 2)... 0 (all)
        '''
        self.x = x
        self.y = y
                
        self.data = []
        for i in range(layers_needed):
            self.data.append(None)
        
        # associate the pixel with its object
        self.net = None
        self.pad = None
        self.wire = None
    
    def setLayerOccupancy(self, layer, occupancy_state):
        self.data[layer - 1] = occupancy_state
    
    def getOccupancy(self):
        return self.data
        


def populatePixels():
    '''
    Fills in the layers list with Pixel objects covering the entire board area.
    Each Pixel object contains occupancy information for each layer.
    '''
    global board_width, board_height

    # get the width and height of the board from the boundary
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    for point in boundary:
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]
    board_width = max_x - min_x
    board_height = max_y - min_y
    print(f"Board dimensions: {board_width} x {board_height}")


    for y in range(int(board_height / PIXEL_SIZE)):
        layers.append([])
        for x in range(int(board_width / PIXEL_SIZE)):
            pixel = Pixel(x * PIXEL_SIZE, y * PIXEL_SIZE)
            for layer in range(layers_needed):
                pixel.setLayerOccupancy(layer + 1, "None")  # default to None
            layers[y].append(pixel)


def updatePixelOccupancy():
    '''
    Update pixel object occupancy based on pad positions and shapes.
    1. Reset all pixel occupancies to None.
    2. Update occupancy based on pad shapes and positions.
    '''

    # reset all pixel occupancy to None
    for y in range(len(layers)):
        for x in range(len(layers[0])):
            pixel = layers[y][x]
            for layer in range(layers_needed):
                pixel.setLayerOccupancy(layer + 1, "None")
    
    # reconstruct pixel occupancy based on pads
    for pad in pads:
        pos = pad.getPosition()
        shape = pad.shape
        shape_type = pad.shape_type
        layers_of_pad = pad.getLayers()  

        # change the occupancy of the pixel at the center of the pad
        pixel_x = int(pos[0] / PIXEL_SIZE)
        pixel_y = int(-pos[1] / PIXEL_SIZE) 

        if shape_type == "circle":
            diameter = shape[0][1]
            radius_in_pixels = int((diameter / 2) / PIXEL_SIZE)
            # set all pixels within the radius to occupied by pad
            for dy in range(-radius_in_pixels, radius_in_pixels + 1):
                for dx in range(-radius_in_pixels, radius_in_pixels + 1):
                    if dx**2 + dy**2 <= radius_in_pixels**2:
                        px = pixel_x + dx
                        py = pixel_y + dy
                        if px < 0 or px >= len(layers[0]) or py < 0 or py >= len(layers):
                            continue
                        pixel = layers[py][px]
                        for layer in layers_of_pad:
                            pixel.setLayerOccupancy(layer, "pad")
                            pixel.net = pad.getNet()
                            pixel.pad = pad
        
        elif shape_type == "polygon":
            # find the bounding box of the polygon
            xs = [vertex[0] + pos[0] for vertex in shape]
            ys = [vertex[1] + pos[1] for vertex in shape]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            min_pixel_x = int(min_x / PIXEL_SIZE)
            max_pixel_x = int(max_x / PIXEL_SIZE)
            min_pixel_y = int(-max_y / PIXEL_SIZE)
            max_pixel_y = int(-min_y / PIXEL_SIZE)

            # for each pixel in the bounding box, check if it's inside the polygon
            for py in range(min_pixel_y, max_pixel_y + 1):
                for px in range(min_pixel_x, max_pixel_x + 1):
                    # convert pixel center to world coordinates
                    world_x = px * PIXEL_SIZE + PIXEL_SIZE / 2
                    world_y = - (py * PIXEL_SIZE + PIXEL_SIZE / 2)

                    # use ray-casting algorithm to check if point is inside polygon
                    inside = False
                    n = len(shape)
                    for i in range(n):
                        v1 = (shape[i][0] + pos[0], shape[i][1] + pos[1])
                        v2 = (shape[(i + 1) % n][0] + pos[0], shape[(i + 1) % n][1] + pos[1])
                        if ((v1[1] > world_y) != (v2[1] > world_y)) and \
                           (world_x < (v2[0] - v1[0]) * (world_y - v1[1]) / (v2[1] - v1[1]) + v1[0]):
                            inside = not inside

                    if inside:
                        if px < 0 or px >= len(layers[0]) or py < 0 or py >= len(layers):
                            continue
                        pixel = layers[py][px]
                        for layer in layers_of_pad:
                            pixel.setLayerOccupancy(layer, "pad")
                            pixel.net = pad.getNet()
                            pixel.pad = pad

            
        #print(pos, shape, shape_type, layers_of_pad) 
    
    for wire in wires:
        if isinstance(wire, Wire):
            segments = wire.getSegments()
            print(f"Processing wire with {segments}")
            for segment in segments:
                p1 = segment[0]
                p2 = segment[1]
                layer = segment[2]
                width = segment[3]

                # Bresenham's line algorithm to find pixels along the line
                x1 = int(p1[0] / PIXEL_SIZE)
                y1 = int(-p1[1] / PIXEL_SIZE)
                x2 = int(p2[0] / PIXEL_SIZE)
                y2 = int(-p2[1] / PIXEL_SIZE)

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                half_width = int(width / (2 * PIXEL_SIZE))

                # Store all pixels along the centerline first
                centerline_pixels = []
                x_temp, y_temp, err_temp = x1, y1, err
                while True:
                    centerline_pixels.append((x_temp, y_temp))
                    if x_temp == x2 and y_temp == y2:
                        break
                    err2 = err_temp * 2
                    if err2 > -dy:
                        err_temp -= dy
                        x_temp += sx
                    if err2 < dx:
                        err_temp += dx
                        y_temp += sy
                
                # Now mark all pixels within half_width of the centerline
                for cx, cy in centerline_pixels:
                    # Mark the centerline pixel
                    if 0 <= cx < len(layers[0]) and 0 <= cy < len(layers):
                        pixel = layers[cy][cx]
                        pixel.setLayerOccupancy(layer, "trace")
                        pixel.net = wire.getNet()
                        pixel.wire = wire
                    
                    # Mark pixels in a square band around the centerline
                    for dx_off in range(-half_width, half_width + 1):
                        for dy_off in range(-half_width, half_width + 1):
                            px = cx + dx_off
                            py = cy + dy_off
                            if 0 <= px < len(layers[0]) and 0 <= py < len(layers):
                                pixel = layers[py][px]
                                pixel.setLayerOccupancy(layer, "trace")
                                pixel.net = wire.getNet()
                                pixel.wire = wire


def displayGrid():
    pixel_maps = [] # each layer gets its own pixel map which is a 2D array

    for layer in range(layers_needed):
        img = np.zeros((int(board_height), int(board_width))).astype(np.uint8)

        # for each pixel, if this layer is occupied by a pad, set the pixel to white
        for y in range(len(layers)):
            for x in range(len(layers[0])):
                pixel = layers[y][x]
                occupancy = pixel.getOccupancy()[layer]
                if "pad" in occupancy:
                    img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = 255
                if "trace" in occupancy:
                    img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = 128

        pixel_maps.append(img)

    # plot each pixel map
    for i in range(len(pixel_maps)):
        img = pixel_maps[i]
        plt.figure()
        plt.title(f"Layer {i + 1}")
        plt.imshow(img)
        plt.axis('off')

    plt.show()


def addClearance(img):
    """Return a copy of img where the -1 region has been expanded by 1 pixel
    in all 8 directions.  If expansion overlaps 1s, they are overwritten by -1.
    """
    img = img.copy()
    neg = (img == -1)

    # 3x3 structuring element -> expand by 1 pixel in all directions
    struct = np.ones((3,3), dtype=bool)
    dilated = binary_dilation(neg, structure=struct)

    img[dilated] = -1
    return img

def retracePath(pixel_map, goal):
    '''
    goal: (x, y, layer)
    retrace the path from goal to start by following the lowest cost neighbour to 1
    '''
    x = goal[0]
    y = goal[1]
    layer = goal[2]

    path = []
    path.append((x, y, layer))
    while pixel_map[layer][y][x] != 1:
        # neighbours =   [(x-1, y-1), (x, y-1), (x+1, y-1),
        #                 (x-1, y),             (x+1, y),
        #                 (x-1, y+1), (x, y+1), (x+1, y+1)]
        neighbours =   [(x, y-1), (x-1, y), (x+1, y), (x, y+1),         # up down left right                  
                        (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)] # diagonals
        
        min_cost = float('inf')
        next_pixel = None

        for neighbour in neighbours:
            nx = neighbour[0]
            ny = neighbour[1]

            # check if neighbour is within bounds
            if nx < 0 or nx >= len(pixel_map[0][0]) or ny < 0 or ny >= len(pixel_map[0]):
                continue

            cost = pixel_map[layer][ny][nx]
            if cost > 0 and cost < min_cost:
                min_cost = cost
                next_pixel = (nx, ny)
        
        if next_pixel is None:
            print("No path found during retrace.")
            return path
        
        x = next_pixel[0]
        y = next_pixel[1]
        path.append((x, y, layer))
    
    # remove intermediate points to only keep waypoints where direction changes
    simplified_path = []
    simplified_path.append(path[0])
    for i in range(1, len(path) - 1):
        p_prev = path[i - 1]
        p_curr = path[i]
        p_next = path[i + 1]

        dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        if dir1 != dir2:
            simplified_path.append(p_curr)
    simplified_path.append(path[-1])

    return simplified_path

def Astar(goal_layer, pixel_map):
    '''
    start: (x, y, layer)
    goal: (x, y, layer)
    pixel_map: [[layer1_pixels], [layer2_pixels], ...] where each layer_pixels is a 2D array
        a 0 means you can route there, a -1 means you can't route there a 1 means it's in the same net and can be routed freely, -2 means it's a waypoint
    return: pixel_map with the path marked
    '''

    '''
    atm, it sticks to the layer of what the goal is on.
    '''

    # for each pixel:
    #   go through every pixel and find the 1s
    #   add 1s loctions to "search_space" list
    # for each pixel in search_space:
    #   find 8 neighbours
    #   if neighbour value is 0, increment cost by 1
    #   add to "next_search_space" list
    # search_space = next_search_space
    # clear next_search_space
    # repeat until goal is found or search_space is empty

    search_space = []
    next_search_space = []
    goal_layer -= 1

    # add all 1s to search_space (vectorized using numpy where/argwhere)
    for layer_idx, layer_arr in enumerate(pixel_map):
        # Use numpy to find indices of ones. argwhere returns rows as (y, x).
        ones = np.argwhere(layer_arr == 1)
        if ones.size:
            # convert to (x,y,layer) tuples and extend the search_space list
            search_space.extend([(int(x), int(y), layer_idx) for y, x in ones.tolist()])
    
    for i in range(1000):
        #print(f"Iteration {i}, search space size: {len(search_space)}")
        for pixel in search_space:
            x = pixel[0]
            y = pixel[1]
            layer = pixel[2]

            # find 8 neighbours
            # neighbours =   [(x-1, y-1, layer), (x, y-1, layer), (x+1, y-1, layer),
            #                 (x-1, y,   layer),                  (x+1, y,   layer),
            #                 (x-1, y+1, layer), (x, y+1, layer), (x+1, y+1, layer)]
            neighbours =   [(x-1, y-1, goal_layer), (x, y-1, goal_layer), (x+1, y-1, goal_layer),
                            (x-1, y,   goal_layer),                       (x+1, y,   goal_layer),
                            (x-1, y+1, goal_layer), (x, y+1, goal_layer), (x+1, y+1, goal_layer)]

            for neighbour in neighbours:
                nx = neighbour[0]
                ny = neighbour[1]
                nlayer = neighbour[2]

                # check if neighbour is within bounds
                if nx < 0 or nx >= len(pixel_map[0][0]) or ny < 0 or ny >= len(pixel_map[0]):
                    continue

                # if neighbour value is 0, increment cost by 1
                if pixel_map[nlayer][ny][nx] == 0:
                    pixel_map[nlayer][ny][nx] = pixel_map[layer][y][x] + 1
                    next_search_space.append((nx, ny, nlayer))
                elif pixel_map[nlayer][ny][nx] == -2:
                    # reached goal
                    print("Goal reached!")
                    path = retracePath(pixel_map, (nx, ny, nlayer))
                    print("Path:", path)
                    return path
        search_space = next_search_space
        next_search_space = []
    
    print("Goal not reached after 1000 iterations.")
    return None
    


def routePads(pad_1, pad_2):

    pos1 = pad_1.getPosition()
    pos2 = pad_2.getPosition()

    print(f"Routing trace between pad {pad_1.getID()} at {pos1} and pad {pad_2.getID()} at {pos2}")
    print(f"pad_1 pos: {pad_1.getPosition()}, layers: {pad_1.getLayers()}")

    net = pad_1.getNet()

    # check if the first waypoint is closer to pad1 or pad2
    if waypoints:
        print(pad_1.getPosition(), waypoints[0])
        print(pad_2.getPosition(), waypoints[0])
        dx1 = pad_1.getPosition()[0] - waypoints[0][0]
        dy1 = -pad_1.getPosition()[1] - waypoints[0][1]
        dist1 = np.sqrt(dx1**2 + dy1**2)
        dx2 = pad_2.getPosition()[0] - waypoints[0][0]
        dy2 = -pad_2.getPosition()[1] - waypoints[0][1]
        dist2 = np.sqrt(dx2**2 + dy2**2)

        print(f"dist1, dist2 = {dist1}, {dist2}")

        if dist1 > dist2:
            # the distance from pad_2 to the first waypoint is closer than the distance from pad_1 to the first waypoint
            waypoints.reverse()
    
    # add pad_2 as a waypoint so we can iterate through each waypoint.
    waypoints.append((pad_2.getPosition()[0], -pad_2.getPosition()[1], pad_2.getLayers()[0]))
    

    for waypoint in waypoints:
        # 1st, construct the pixel map where 0 means you can't route there.
        pixel_maps = [] # each layer gets its own pixel map which is a 2D array

        print(f"waypoints: {waypoints}")
        print(f"waypoint: {waypoint}, {int(waypoint[0])}, {int(waypoint[1])}")

        for layer in range(layers_needed):
            img = np.zeros((int(board_height), int(board_width)))

            # for each pixel, if this layer is occupied by a pad which is not pad_1 or pad_2, set the pixel to 0
            for y in range(len(layers)):
                for x in range(len(layers[0])):
                    pixel = layers[y][x]
                    occupancy = pixel.getOccupancy()[layer]
                    
                    # not allowed to route over currently occupied pixels except if the pixel is in the pad or wire in the net
                    if ("pad" in occupancy and pixel.pad != pad_1) or ("trace" in occupancy and pixel.wire.net != net):
                        img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = -1

                    elif ("pad" in occupancy and pixel.pad == pad_1):
                        img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = 1
                    
                    if "trace" in occupancy and (pad_1 == pixel.wire.pad_1 or pad_1 == pixel.wire.pad_2):
                        img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = 1

                    if x == int(waypoint[0]) and y == int(waypoint[1]) and layer + 1 == waypoint[2]:
                        img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = -2
                    
                    # if it's the last waypoint (end pad), set entire pad as destination
                    if waypoints.index(waypoint) + 1 == len(waypoints):
                        if ("pad" in occupancy and pixel.pad == pad_2):
                            print(f"last pad and pixel pad at {x}, {y}")
                            img[int(y * PIXEL_SIZE):int((y + 1) * PIXEL_SIZE), int(x * PIXEL_SIZE):int((x + 1) * PIXEL_SIZE)] = -2
                            
            # binary morphological dilation to add clearance around other non-routable pixels
            img = addClearance(img)

            pixel_maps.append(img)
        
        if ROUTE_DEBUG:
            # plot each pixel map
            for i in range(len(pixel_maps)):
                img = pixel_maps[i]
                plt.figure()
                plt.title(f"Layer {i + 1}")
                plt.imshow(img)
                plt.axis('off')
            plt.show()

        path = Astar(waypoints[0][2], pixel_maps)
        # create a Wire object from the path
        if path is not None:
            segments = []
            for i in range(len(path) - 1):
                p1 = (path[i][0] / PIXEL_SIZE, -path[i][1] / PIXEL_SIZE)
                p2 = (path[i + 1][0] / PIXEL_SIZE, -path[i + 1][1] / PIXEL_SIZE)
                layer = path[i][2] + 1
                width = 1
                segments.append((p1, p2, layer, width))
            print(segments)
            
            wire = Wire(net, segments)
            wire.pad_1 = pad_1
            wire.pad_2 = pad_2
            wires.append(wire)
            updatePixelOccupancy()
        
        else:
            # A star failed - could not route
            return False
        
        if ROUTE_DEBUG:
            displayGrid()

    return True



def orderWires():
    '''
    create a list showing how all the pads connect together (don't care about what net) and ordered based on length
    [(pad1, pad2), (pad3, pad4), ...]
    Where the distance between pad1 and pad2 is less than the distance between pad3 and pad4, etc.
    '''
    pad_pairs = []
    seen_pairs = set()
    for net in nets:
        points = net.getPoints() # points are [x,y] positions of pads and vias

        # find the closest pair of points for each point
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
                # find the pads that correspond to the two points
                pad1 = next((pad for pad in pads if pad.getPosition() == point), None)
                pad2 = next((pad for pad in pads if pad.getPosition() == closest_point), None)
                if pad1 is None or pad2 is None:
                    continue

                # canonicalize pair order by pad ID to avoid duplicates (pad1,pad2) and (pad2,pad1)
                id1 = pad1.getID()
                id2 = pad2.getID()
                if id1 == id2:
                    continue
                pair_key = (min(id1, id2), max(id1, id2))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # append pads in canonical order (lower id first)
                if id1 <= id2:
                    pad_pairs.append((pad1, pad2, dist))
                else:
                    pad_pairs.append((pad2, pad1, dist))
    # sort pad pairs based on distance
    pad_pairs.sort(key=lambda x: x[2])
    print(pad_pairs)

    return pad_pairs


def next_trace_to_route(pad_pairs, skip=False):

    if len(pad_pairs) == 0:
        return None, None
    
    if not skip:
        pad1, pad2, dist = pad_pairs.pop(0)
    else:
        # rotate the list
        pad_pairs.append(pad_pairs.pop(0))
        pad1, pad2, dist = pad_pairs[0]
    return pad1, pad2
        

if __name__ == "__main__":
    processDSNfile("C:/Users/zacap/Documents/projects/CSHackathon2025/ezpcb/DSN/mosfetDriver.dsn")

    pg.init()
    fps = 60
    WIDTH, HEIGHT = 800, 600
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    running = True

    space = pymunk.Space()
    space.gravity = 0, 0
    space.sleep_time_threshold = 0.3
    space.damping = 0.1

    #draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options = CustomDrawOptions(screen)
    draw_options.flags &= ~SDO.DRAW_CONSTRAINTS
    pymunk.pygame_util.positive_y_is_up = False

    total_time = 0
    zoom = 2.0
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

    # determine how many layers are needed
    layers_needed = 2
    for pad in pads:
        if max(pad.getLayers()) > layers_needed:
            layers_needed = max(pad.getLayers())
    
    PIXEL_SIZE = 1 # can go down to 0.1
    board_width = 0
    board_height = 0

    bodies, shapes, traces, component_to_body, pad_to_spring = addPhysicsObjects(space)

    # mouse drag state for left-click dragging
    grabbed_body = None
    grab_joint = None
    grabbed_was_static = False

    # kinematic body that follows the mouse for dragging
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    space.add(mouse_body)

    placement_done = False
    waypoint_place = False
    auto_route_all = False
    waypoints = [] # list of (x, y, layer) tuples for waypoints
    pad_pairs = []
    pad1, pad2 = None, None
    current_layer = 1

    print("Press 'f' to finish placement and start routing.")

    while running:
        for event in pg.event.get():           
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = event.pos
                # convert to world coords used by bodies
                world_x = (mouse_x - dx) / zoom
                world_y = (mouse_y - dy) / zoom

                if event.button == 3:  # Right mouse button (pan)
                    is_dragging = True
                    offset_x = event.pos[0] - dx
                    offset_y = event.pos[1] - dy

                elif event.button == 1 and not placement_done:  # Left mouse button (select/drag or double-click)
                    current_time = pg.time.get_ticks()

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

                if event.button == 1 and placement_done and waypoint_place:
                    waypoints.append((world_x, world_y, current_layer))
                    print(waypoints)

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
            
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    placement_done = True
                    pad_pairs = orderWires()
                    pad1, pad2 = next_trace_to_route(pad_pairs, True)

                    populatePixels()
                    updatePixelOccupancy()

                    print("Press 'w' to toggle waypoint placement mode.")
                    print("Press 'escape' to exit waypoint placement mode.")
                    print("Press 'n' to move to the next trace to route.")
                if event.key == pg.K_w:
                    waypoint_place = not waypoint_place
                if event.key == pg.K_ESCAPE:
                    waypoint_place = False
                if event.key == pg.K_n and placement_done:
                    pad1, pad2 = next_trace_to_route(pad_pairs, True)
                    waypoints = []
                if event.key == pg.K_z and (pg.key.get_mods() & pg.KMOD_CTRL):
                    if len(waypoints) > 0:
                        waypoints.pop()
                # select current layer for waypoints
                if event.key == pg.K_t:
                    current_layer = 1
                if event.key == pg.K_b:
                    current_layer = 2
                if event.key == pg.K_r:
                    # route the current pads
                    if routePads(pad1, pad2):
                        # remove from pad pairs
                        pad_pairs.pop(0)
                if event.key == pg.K_a:
                    # autoroute all
                    auto_route_all = True
                    
                    if routePads(pad1, pad2):
                        # remove from pad pairs
                        pad_pairs.pop(0)
                


        draw_options.transform = pymunk.Transform.scaling(zoom).translated(dx / zoom, dy / zoom)

        if not placement_done:
            space.step(1.0 / fps)

        # freeze static bodies in place
        for body in static_bodies:
            if disable_body_while_dragging == body:
                continue
            body.mass = 1000000
            body.moment = 1000000
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0
            body.force = (0.0, 0.0)
            body.torque = 0.0
            # snap rotation to nearest 90 degrees
            angle = body.angle
            nearest_90 = round(angle / (math.pi / 2)) * (math.pi / 2)
            body.angle = nearest_90
        
        
        # update component positions based on physics bodies
        for body in bodies:
            # find the component that corresponds to this body
            comp = next((c for c in components if component_to_body.get(c.getID()) == body), None)
            if comp is not None:
                comp.move(body.position.x, -body.position.y, body.angle)
                
        screen.fill("black")

        space.debug_draw(draw_options)

        drawPygameComponents()
        drawTraces()


        # choose which trace is about to be routed
        if placement_done:
            # draw all ratlines in grey
            for pair in pad_pairs:
                p1, p2, dist = pair
                pos1 = p1.getPosition()
                pos2 = p2.getPosition()
                x1 = int(pos1[0] * zoom + dx)
                y1 = int(-pos1[1] * zoom + dy)
                x2 = int(pos2[0] * zoom + dx)
                y2 = int(-pos2[1] * zoom + dy)
                pg.draw.line(screen, "grey", (x1, y1), (x2, y2), 2)

            # draw a line between pad1 and pad2
            if pad1 is not None and pad2 is not None:
                pos1 = pad1.getPosition()
                pos2 = pad2.getPosition()
                x1 = int(pos1[0] * zoom + dx)
                y1 = int(-pos1[1] * zoom + dy)
                x2 = int(pos2[0] * zoom + dx)
                y2 = int(-pos2[1] * zoom + dy)
                pg.draw.line(screen, "green", (x1, y1), (x2, y2), 3)
            
            # render green circle at mouse position
            if waypoint_place:
                if current_layer == 1:
                    colour = "red"
                elif current_layer == 2:
                    colour = "blue"
                else:
                    colour = "yellow"
                pg.draw.circle(screen, colour, (mouse_x, mouse_y), 5, 0)

            for waypoint in waypoints:
                wx = int(waypoint[0] * zoom + dx)
                wy = int(waypoint[1] * zoom + dy)
                layer = waypoint[2]
                if layer == 1:
                    colour = "red"
                elif layer == 2:
                    colour = "blue"
                pg.draw.circle(screen, colour, (wx, wy), 5, 0)

        else:
            # draw springs as white lines
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
                    pg.draw.line(screen, (255, 255, 255), to_pygame(p1, screen), to_pygame(p2, screen), 2)
       
        if auto_route_all:
            if len(pad_pairs) > 0:
                pad1, pad2 = next_trace_to_route(pad_pairs, True)
                waypoints = []
                if routePads(pad1, pad2):
                    # remove from pad pairs
                    pad_pairs.pop(0)
            else:
                auto_route_all = False



        pg.display.flip()
        dt = clock.tick(fps)
        total_time += dt / 1000.0

