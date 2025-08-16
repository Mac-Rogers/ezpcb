import numpy as np

pads = []
nets = []
components = []

class Pad:
    def __init__(self, name, ID, position, shape, outline, layer):
        '''
        name [str]: name of the pad
        ID [int]: ID of the pad
        position [tuple]: (x, y) position of the pad
        shape [str]: shape of the pad (e.g., "circle", "polygon")
        outline [list]: outline of the pad
                        This is a list of points (polygon: [x0 y0 x1 y1 ...], circle: [diamerter]) 
                        that are relative to the pad's position
        layer [int]: layer of the pad (1 is top, 2 is bottom, 21, 22,... are interlayers)
        '''
        self.name = name
        self.ID = int(ID)
        self.position = position
        self.shape = shape
        self.outline = outline
        self.layer = layer

        # if the shape is a circle, the outline is the diameter
        # if the shape is a polygon, the outline is a path which an aperture follows
        # this path is relative to self.position
    
    def getName(self):
        return self.name
    
    def getID(self):
        return self.ID
    
    def getPosition(self):
        return self.position

    def addShape(self, shape, outline, layer):
        self.shape = shape
        self.outline = outline
        self.layer = layer

class Net:
    def __init__(self, name, pads):
        '''
        name [str]: name of the net
        pads [list]: list of pad objects connected to the net
        '''
        self.name = name
        self.pads = pads
        self.wires = [] # will contain tuples of coordinates for wire segments in the net, and layer number
        self.vias = []

    def addWireSegment(self, start, end, layer):
        '''
        A wire segment is straight line between 2 points start:(x1, y1) and end:(x2, y2)
        '''
        self.wires.append((start, end, layer))
    
    def addVia(self, position):
        self.vias.append(position)

    def getPadsInNet(self):
        return self.pads


class Component:
    def __init__(self, pad):
        '''
        pad is a Pad object that belongs to the component
        '''
        # component has pad
        # component has bounding box
        # component has clearance
        # component has centre location

        # component can be moved -> moves pads
        # component can be rotated -> moves pads
        self.pads = [] # list of pad objects
        

    def addPad(self, pad):
        self.pads.append(pad)
    
boundary = []

def processDSNfile(file_name):
    # function to populate the pads and nets lists from a DSN file

    global boundary

    with open(file_name, "r") as file:
        content = file.read()
        #print(content)


    lines = content.split("\n")
    #spaces = []

    for i in range(len(lines)):
        line = lines[i]

        # conditioned line to remove the first and last bracket
        line = line.strip()[1:-1]

        if "boundary" in line:
            boundary = line.split(" ")[3:]
            if boundary[-1] == '':
                boundary.pop()  # remove last empty string if it exists
            for i in range(len(boundary)):
                boundary[i] = float(boundary[i])

        # may want to handle line "via via0 via1"

        if "pin " in line:
            pinInfo = line.split(" ")
            pinName = pinInfo[1]
            pinID = pinInfo[2]
            pinPosition = float(pinInfo[3]), float(pinInfo[4])

            pads.append(Pad(pinName, pinID, pinPosition, None, None, None))
        
        if "shape" in line:
            
            for pad_i in (pads):
                if pad_i.getName() in lines[i - 1] or pad_i.getName() in lines[i - 2]:
                    shape_description = lines[i].strip()[1:-2].split("(")[1].split(" ")
                    shape_type = shape_description[0]
                    shape_layer = shape_description[1]
                    # aperture is shape_description[2]

                    if shape_type == "polygon":
                        shape_outline = shape_description[3:]
                        for j in range(len(shape_outline)):
                            shape_outline[j] = float(shape_outline[j])

                    if shape_type == "circle":
                        shape_outline = shape_description[2]

                    pad_i.addShape(shape_type, shape_outline, shape_layer)
        
        if "pins" in line:
            # pins_in_net is a list of the IDs of the pads
            pins_in_net = line.split(" ")[1:]
            for j in range(len(pins_in_net)):
                pins_in_net[j] = int(pins_in_net[j].split("-")[1])
            
            # pads_in_net is a list of the pad objects in the net
            #print(pins_in_net, pads)
            pads_in_net = []
            for pin in pins_in_net:
                for pad in pads:
                    #print(f"padid: {type(pad.getID())}, {type(pin)}")
                    if pad.getID() == pin:
                        pads_in_net.append(pad)
            #print(f"pads_in_net: {pads_in_net}")
            # net name from previous line
            net_name = lines[i - 1].strip().split(" ")[1]

            nets.append(Net(net_name, pads_in_net))
    
    # sort pads into components based on ID
    # if two consecutive pins have a difference in ID by < 16, they belong to the same component
    last_pad_id = 0
    for pad in pads:
        if pad.getID() - last_pad_id < 16:
            # add pad to current component
            #components.append(Component())
            pass
        else:
            # create new component
            pass
        last_pad_id = pad.getID()


def processSESfile(file_name):
    # function to create a .ses file from the new wires and vias

    # write to a new .ses file
    with open(file_name, "w") as file:
        # header
        file.write(f"(session \"{file_name}\"\n\t(routes\n\t\t(resolution mil 1000)\n\t\t(network_out\n")

        for net in nets:
            file.write(f"\t\t\t(net {net.name}\n")
            #print(f"wire segments: {net.wires}")
            #print(len(net.wires))
            for i in range(len(net.wires)):
                #print(i)
                #print(f"wire segment {i}: {net.wires[i][0]}, {net.wires[i][1]}")
                file.write(f"\t\t\t\t(wire\n\t\t\t\t\t(path {net.wires[i][2]} 1000\n")
                file.write(f"\t\t\t\t\t\t{net.wires[i][0][0]} {net.wires[i][0][1]}\n")
                file.write(f"\t\t\t\t\t\t{net.wires[i][1][0]} {net.wires[i][1][1]}\n")
                file.write(f"\t\t\t\t\t)\n\t\t\t\t)\n")
            
            for i in range(len(net.vias)):
                file.write(f"\t\t\t\t(via via0 {net.vias[i][0]} {net.vias[i][1]})\n")

            file.write(f"\t\t\t)\n")

        file.write(f"\t\t)\n\t)\n)\n")

def convertCoordinates(coords):
    # coords is a tuple of (x, y)
    x, y = coords
    return (x * 1000, y * 1000)

def veryBasicRoute():

    for net in nets:
        pads_in_net = net.getPadsInNet()
    
        for i in range(len(pads_in_net) - 1):
            #print(f"net: {net.name}, {pad.getPosition()}")
            pad1_pos = pads_in_net[i].getPosition()
            pad2_pos = pads_in_net[i + 1].getPosition()

            pad1_pos = convertCoordinates(pad1_pos)
            pad2_pos = convertCoordinates(pad2_pos)

            #print(f"net: {net.name}, {pad1_pos} to {pad2_pos}")
            net.addWireSegment(pad1_pos, pad2_pos, 1)

def makeBoundaryAReasonableFormat(boundary):
    # currently formatted as: [x1 y1 x2 y2 ... xn yn]
    xs = boundary[0::2]
    ys = boundary[1::2]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return ((0,0), (width * 1000, height * 1000))

def occupancyGrid():
    grid_spacing = 1000 # measured in 1/1000 thou or something
    corners = makeBoundaryAReasonableFormat(boundary)
    print(corners)
    number_of_cells_x = corners[1][0] // grid_spacing
    number_of_cells_y = corners[1][1] // grid_spacing
    print(f"Number of cells: {number_of_cells_x} x {number_of_cells_y}")

    # check if pad overlaps the corner of a grid
    



processDSNfile("DSN/basic1layerRoute.dsn")

occupancyGrid()
veryBasicRoute()
nets[1].addWireSegment((0,0), (94.6*1000,27.5*1000), 1)


processSESfile("SES/basic1layerRoute.ses")
