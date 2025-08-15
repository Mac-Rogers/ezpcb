import numpy as np

pads = []

class pad:
    def __init__(self, name, ID, position, shape, outline, layer):
        self.name = name
        self.ID = ID
        self.position = position
        self.shape = shape
        self.outline = outline
        self.layer = layer
    
    def getName(self):
        return self.name

    def addShape(self, shape, outline, layer):
        self.shape = shape
        self.outline = outline
        self.layer = layer

with open("Autorouter_PCB_cs-hackathon_2025-08-15 (2).dsn", "r") as file:
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

        pads.append(pad(pinName, pinID, pinPosition, None, None, None))
    
    if "shape" in line:
        
        for pad_i in (pads):
            if pad_i.getName() in lines[i - 1] or pad_i.getName() in lines[i - 2]:
                shape_description = lines[i].strip()[1:-2].split("(")[1].split(" ")
                shape_type = shape_description[0]
                shape_layer = shape_description[1]
                # aperture is shape_description[2]

                if shape_type == "polygon":
                    shape_outline = shape_description[3:]
                    for point in shape_outline:
                        point = float(point)
                
                if shape_type == "circle":
                    shape_outline = shape_description[2]

                #print(pad_i.getName(), shape_description)
                pad_i.addShape(shape_type, shape_outline, shape_layer)
                # the previous line is a 

#print(spaces)
#print(boundary)

for pad_i in pads:
    print(f"Pad Name: {pad_i.getName()}, ID: {pad_i.ID}, Position: {pad_i.position}, Shape: {pad_i.shape}, Outline: {pad_i.outline}, Layer: {pad_i.layer}")
