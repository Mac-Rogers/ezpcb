import numpy as np
import pygame as pg

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

class Component:
    def __init__(self, id, comp_pads):
        self.id = id
        self.pads = comp_pads  # list of pad objects

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

    ######################### LIBRARY #########################

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
                pg.draw.circle(screen, color, (int(x * zoom + pan_x), int(y * zoom + pan_y)), int(diameter/2 * zoom), 0)
            elif pad.shape_type == "polygon":
                points = []
                for vertex in pad.shape:
                    vx = int((vertex[0] + pad.getPosition()[0]) * zoom + pan_x)
                    vy = int((-vertex[1] - pad.getPosition()[1]) * zoom + pan_y)
                    points.append((vx, vy))
                pg.draw.polygon(screen, color, points, 0)
        
        p1, p2 = component.getBoundingBox()
        #print(p1, p2)
        if p1 is not None and p2 is not None:
            x1 = int(p1[0] * zoom + pan_x)
            y1 = int(p1[1] * zoom + pan_y)
            x2 = int(p2[0] * zoom + pan_x)
            y2 = int(p2[1] * zoom + pan_y)
            pg.draw.rect(screen, "yellow", (x1, y1, x2 - x1, y2 - y1), 1)

    # draw the boundary
    if len(boundary) > 1:
        boundary_points = []
        for point in boundary:
            bx = int(zoom * point[0] + pan_x)
            by = int(zoom * -point[1] + pan_y)
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
                x1 = int(p1[0] * zoom + pan_x)
                y1 = int(-p1[1] * zoom + pan_y)
                x2 = int(closest_point[0] * zoom + pan_x)
                y2 = int(-closest_point[1] * zoom + pan_y)
                pg.draw.line(screen, "white", (x1, y1), (x2, y2), 1)

def getComponentAt(x, y):
    for component in components:
        p1, p2 = component.getBoundingBox()
        if p1 is not None and p2 is not None:
            x1 = int(p1[0] * zoom + pan_x)
            y1 = int(p1[1] * zoom + pan_y)
            x2 = int(p2[0] * zoom + pan_x)
            y2 = int(p2[1] * zoom + pan_y)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return component
    return None

screen_width = 800
screen_height = 600

pan_x = 0
pan_y = 0
zoom = 1.0
pan_draging = False
component_draging = False
current_component = None
mouse_x = 0
mouse_y = 0

if __name__ == "__main__":
    processDSNfile("DSN/mosfetDriver.dsn")

    printStructure()

    pg.init()
    screen = pg.display.set_mode((screen_width, screen_height))
    clock = pg.time.Clock()
    running = True

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            # the following is for dragging:
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right mouse button
                    pan_draging = True
                    mouse_x, mouse_y = event.pos
                    offset_x = pan_x - mouse_x
                    offset_y = pan_y - mouse_y
                
                if event.button == 1:  # Left mouse button
                    component_draging = True
                    mouse_x, mouse_y = event.pos
                    current_component = getComponentAt(mouse_x, mouse_y)

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 3:  # Right mouse button
                    pan_draging = False

                if event.button == 1:  # Left mouse button
                    component_draging = False
                    current_component = None

            elif event.type == pg.MOUSEMOTION:
                if pan_draging:
                    mouse_x, mouse_y = event.pos
                    pan_x = mouse_x + offset_x
                    pan_y = mouse_y + offset_y
                
                if component_draging and current_component is not None:
                    mouse_x, mouse_y = event.pos
                    # Convert screen coordinates to world coordinates by accounting for zoom
                    offset_x = (mouse_x - current_component.getPos()[0] * zoom - pan_x) / zoom
                    offset_y = (-mouse_y + current_component.getPos()[1] * zoom + pan_y) / zoom
                    current_component.move(offset_x, offset_y)
            
            # scrolling to zoom:
            elif event.type == pg.MOUSEWHEEL:
                if event.y > 0:  # scroll up
                    zoom *= 1.1
                elif event.y < 0:  # scroll down
                    zoom /= 1.1
                



        screen.fill("black")

        drawComponents()
        drawNets()

        pg.display.flip()
        clock.tick(60)