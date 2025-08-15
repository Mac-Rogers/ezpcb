import numpy as np

pads = []
nets = []

class Pad:
    def __init__(self, name, ID, position, shape, outline, layer):
        self.name = name
        self.ID = ID
        self.position = position
        self.shape = shape
        self.outline = outline
        self.layer = layer

        # if the shape is a circle, the outline is the diameter
        # if the shape is a polygon, the outline is a path which an aperture follows
        # this path is relative to self.position
    
    def getName(self):
        return self.name

    def addShape(self, shape, outline, layer):
        self.shape = shape
        self.outline = outline
        self.layer = layer

class Net:
    def __init__(self, name, pins):
        self.name = name
        # pins is an array
        self.pins = pins
        self.wires = [] # will contain tuples of coordinates for wire segments in the net, and layer number
        self.vias = []

    def addWireSegment(self, start, end, layer):
        self.wires.append((start, end, layer))
    
    def addVia(self, position):
        self.vias.append(position)



def processDSNfile(file_name):

    with open(file_name, "r") as file:
        content = file.read()
        #print(content)


    lines = content.split("\n")
    #spaces = []
    boundary = []

    for i in range(len(lines)):
        line = lines[i]

        # count leading spaces
        '''num_spaces = len(line) - len(line.lstrip(' '))
        if num_spaces % 2 == 1:
            num_spaces -= 1
        spaces.append(num_spaces)
        '''
        # conditioned line to remove the first and last bracket
        line = line.strip()[1:-1]

        if "boundary" in line:
            boundary = line.split(" ")[2:]
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
            pins_in_net = line.split(" ")[1:]
            for j in range(len(pins_in_net)):
                pins_in_net[j] = int(pins_in_net[j].split("-")[1])

            # net name from previous line
            net_name = lines[i - 1].strip().split(" ")[1]

            nets.append(Net(net_name, pins_in_net))


def processSESfile(file_name):
    # write to a new .ses file
    with open(file_name, "w") as file:
        # header
        file.write(f"(session \"{file_name}\"\n\t(routes\n\t\t(resolution mil 1000)\n\t\t(network_out\n")

        for net in nets:
            file.write(f"\t\t\t(net {net.name}\n")
            print(f"wire segments: {net.wires}")
            #print(len(net.wires))
            for i in range(len(net.wires)):
                print(i)
                print(f"wire segment {i}: {net.wires[i][0]}, {net.wires[i][1]}")
                file.write(f"\t\t\t\t(wire\n\t\t\t\t\t(path {net.wires[i][2]} 1000\n")
                file.write(f"\t\t\t\t\t\t{net.wires[i][0][0]} {net.wires[i][0][1]}\n")
                file.write(f"\t\t\t\t\t\t{net.wires[i][1][0]} {net.wires[i][1][1]}\n")
                file.write(f"\t\t\t\t\t)\n\t\t\t\t)\n")

            file.write(f"\t\t\t)\n")

        file.write(f"\t\t)\n\t)\n)\n")


processDSNfile("Autorouter_PCB_cs-hackathon_2_2025-08-15.dsn")

nets[0].addWireSegment((0, 0), (100000, 100000), 1)
nets[0].addWireSegment((100000, 100000), (200000, 100000), 2)
nets[1].addWireSegment((0, 0), (100000, 50000), 1)

processSESfile("Autorouter_PCB_cs-hackathon_2_2025-08-15.ses")

'''
for pad_i in pads:
    print(f"Pad Name: {pad_i.getName()}, ID: {pad_i.ID}, Position: {pad_i.position}, Shape: {pad_i.shape}, Outline: {pad_i.outline}, Layer: {pad_i.layer}")
print('')
for net in nets:
    print(f"Net Name: {net.name}, Pins: {net.pins}")
    '''