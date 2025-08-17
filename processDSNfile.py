import numpy as np
import math
import random
from collections import deque
import sys
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QPlainTextEdit, QSizePolicy, QSpacerItem
)
from rectpack import newPacker
import matplotlib.pyplot as plt
import matplotlib.patches as patches


pads = []
nets = []
components = []

GRID_SPACING = 2
FILE_NAME = "basic1layerRoute"

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
        self.centre_offset = (0,0)

        self.occupancy_grid_position= []

        # if the shape is a circle, the outline is the diameter
        # if the shape is a polygon, the outline is a path which an aperture follows
        # this path is relative to self.position
    def getId(self):
        return self.ID

    def getName(self):
        return self.name
    
    def getID(self):
        return self.ID
    
    def setPosition(self, x, y):
        self.position = (x, y)
    
    def getPosition(self):
        return self.position

    def setOffset(self, x, y):
        self.centre_offset = (x, y)

    def getOffset(self):
        return self.centre_offset

    def addShape(self, shape, outline, layer):
        self.shape = shape
        self.outline = outline
        self.layer = layer
    
    def getVertices(self):
        vertices = []
        if self.shape == "circle":
            # For a circle, the outline is just the diameter
            diameter = self.outline[0]
            radius = diameter / 2
            center = self.position
            # Create a circle approximation with 8 points
            for angle in range(0, 360, 45):
                x = center[0] + radius * np.cos(np.radians(angle))
                y = center[1] + radius * np.sin(np.radians(angle))
                vertices.append((x, y))
        elif self.shape == "polygon":
            # For a polygon, the outline is a list of points
            for i in range(0, len(self.outline), 2):
                x = self.position[0] + self.outline[i]
                y = self.position[1] + self.outline[i + 1]
                vertices.append((x, y))
        return vertices

    def addOccupancyGridPosition(self, column, row):
        self.occupancy_grid_position.append((column, row))


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
        self.occupancy_grid_position = []  # list of tuples (column, row) for occupancy grid

    def addWireSegment(self, start, end, layer):
        '''
        A wire segment is straight line between 2 points start:(x1, y1) and end:(x2, y2)
        '''
        self.wires.append((start, end, layer))
    
    def addVia(self, position):
        self.vias.append(position)

    def getPadsInNet(self):
        return self.pads

    def getWiresInNet(self):
        return self.wires

    def addOccupancyGridPosition(self, column, row):
        self.occupancy_grid_position.append((column, row))

    def getComponents(self):
        ans = []
        for pad in self.pads:
            for component in components:
                if pad in component.pads:
                    ans.append(component)
        return ans

    def getLength(self):
        '''
        Compute the minimal spanning tree length of all pads in this net using Prim's algorithm
        Returns the total length of the MST
        '''
        if len(self.pads) <= 1:
            return 0.0
        
        # Get all pad positions
        positions = [pad.getPosition() for pad in self.pads]
        n = len(positions)
        
        # Distance function (Euclidean distance)
        def distance(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Prim's algorithm
        visited = [False] * n
        min_edge = [float('inf')] * n
        parent = [-1] * n
        
        # Start from the first pad
        min_edge[0] = 0
        total_length = 0.0
        
        for _ in range(n):
            # Find minimum edge
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_edge[v] < min_edge[u]):
                    u = v
            
            visited[u] = True
            if parent[u] != -1:
                total_length += distance(positions[u], positions[parent[u]])
            
            # Update minimum edges for adjacent vertices
            for v in range(n):
                if not visited[v]:
                    edge_weight = distance(positions[u], positions[v])
                    if edge_weight < min_edge[v]:
                        min_edge[v] = edge_weight
                        parent[v] = u
        
        return total_length


class Component:
    def __init__(self, id):
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
        self.position = (0,0)
        self.theta = 0
        self.dimensions = None
        self.id = id

    def setTheta(self, theta):
        self.optimal_theta = theta

    def getTheta(self):
        return self.theta

    def move(self, new_x, new_y):
        current_x, current_y = self.position
        diff_x, diff_y = new_x - current_x, new_y - current_y
        self.position = (new_x, new_y)
        for pad in self.pads:
            pad_x, pad_y = pad.getPosition()
            pad.setPosition(diff_x + pad_x, diff_y + pad_y)

    def setDimensions(self):
        if not self.pads:
            self.dimensions = (0, 0)
            return
            
        minx = float('inf')
        maxx = float('-inf')
        miny = float('inf')
        maxy = float('-inf')
        
        for pad in self.pads:
            pad_x, pad_y = pad.getPosition()
            
            if pad.shape == "circle":
                radius = pad.outline / 2
                minx = min(minx, pad_x - radius)
                maxx = max(maxx, pad_x + radius)
                miny = min(miny, pad_y - radius)
                maxy = max(maxy, pad_y + radius)
                
            elif pad.shape == "polygon":
                vertices = pad.getVertices()
                for vertex_x, vertex_y in vertices:
                    minx = min(minx, vertex_x)
                    maxx = max(maxx, vertex_x)
                    miny = min(miny, vertex_y)
                    maxy = max(maxy, vertex_y)
            else:
                minx = min(minx, pad_x)
                maxx = max(maxx, pad_x)
                miny = min(miny, pad_y)
                maxy = max(maxy, pad_y)
        
        # Calculate total dimensions (width, height)
        width = round(maxx - minx, 2)
        height = round(maxy - miny, 2)
        self.dimensions = (width, height)
        
    def getDimensions(self):
        if self.dimensions is None:
            self.setDimensions()
        return self.dimensions

    def rotate(self, angle):
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        comp_x, comp_y = self.getPosition()
        
        for pad in self.pads:
            pad_x, pad_y = pad.getPosition()

            # Calculate relative position to centre
            rel_x = pad_x - comp_x
            rel_y = pad_y - comp_y

            # Perform rotation
            new_pad_x = comp_x + rel_x * cos_a - rel_y * sin_a
            new_pad_y = comp_y + rel_x * sin_a + rel_y * cos_a

            pad.setPosition(new_pad_x, new_pad_y)


            

    def setCentre(self, x, y):
        self.position = (x,y)   

    def getPosition(self):
        return self.position

    def addPad(self, pad):
        self.pads.append(pad)

class GridTile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.objects = [] # objects can be pad object or wire object or via object
        self.a_star_weight = [None, None] # 2 element array for top and bottom layer
    
    def addObject(self, object):
        self.objects.append(object)



class SliderRow(QWidget):
    """A named horizontal slider with 0.1 resolution and live value label."""
    def __init__(self, name: str, minimum: float = 0.0, maximum: float = 100.0,
                 initial: float = None, step: float = 0.1, parent=None):
        super().__init__(parent)
        if step <= 0:
            raise ValueError("step must be > 0")
        if maximum < minimum:
            minimum, maximum = maximum, minimum

        self.name = name
        self.step = step
        self.scale = int(round(1.0 / step))  # e.g. 0.1 -> 10, 0.05 -> 20
        self.min_float = float(minimum)
        self.max_float = float(maximum)
        if initial is None:
            initial = (self.min_float + self.max_float) / 2.0
        initial = max(self.min_float, min(self.max_float, initial))

        # Widgets
        self.name_lbl = QLabel(str(name))
        self.name_lbl.setMinimumWidth(120)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self._to_int(self.min_float), self._to_int(self.max_float))
        self.slider.setTickInterval(max(1, (self._to_int(self.max_float) - self._to_int(self.min_float)) // 10))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(1)  # 1 int step = `step` in float
        self.slider.setPageStep(2)    # keyboard PgUp/PgDn = 2 steps (~0.2 if step=0.1)
        self.slider.setValue(self._to_int(initial))

        self.value_lbl = QLabel(self._fmt(initial))
        self.value_lbl.setMinimumWidth(60)
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider.valueChanged.connect(self._on_value_changed)

        row = QHBoxLayout(self)
        row.addWidget(self.name_lbl)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_lbl)

    def _to_int(self, x: float) -> int:
        return int(round(x * self.scale))

    def _to_float(self, i: int) -> float:
        return i / float(self.scale)

    def _fmt(self, x: float) -> str:
        # show 1 decimal for 0.1 steps, but trim trailing zeros nicely
        return f"{x:.3f}".rstrip('0').rstrip('.')  # supports other steps like 0.05 too

    def _on_value_changed(self, i: int):
        self.value_lbl.setText(self._fmt(self._to_float(i)))

    # Public API
    def value(self) -> float:
        return self._to_float(self.slider.value())

    def set_value(self, x: float):
        x = max(self.min_float, min(self.max_float, x))
        self.slider.setValue(self._to_int(x))


class SliderPanel(QWidget):
    """
    Stack of named sliders (each with its own range), a Submit button, and a log.
    Add sliders one-by-one with add_slider(...).
    """
    valuesSubmitted = pyqtSignal(list)  # emits list[float] on Submit

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Named Sliders (0.1 resolution) + Log")
        self.resize(560, 480)

        self._sliders = []

        self.layout = QVBoxLayout(self)

        # Spacer keeps sliders compact if many are added
        self.layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Submit button
        self.submit_btn = QPushButton("Submit Values")
        self.submit_btn.clicked.connect(self.on_submit)
        self.layout.addWidget(self.submit_btn)

        # Log
        self.layout.addWidget(QLabel("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Messages will appear here...")
        self.layout.addWidget(self.log, 1)

        self.submit_btn.setAutoDefault(True)
        self.submit_btn.setDefault(True)

        self.log_message("Ready. Add sliders using add_slider(name, min, max, initial, step=0.1).")

    # ---- Slider management ----
    def add_slider(self, name: str, minimum: float, maximum: float, initial: float = None, step: float = 0.1):
        row = SliderRow(name=name, minimum=minimum, maximum=maximum, initial=initial, step=step)
        # Insert before the submit button/spacer section (i.e., near the top)
        # We want all sliders stacked above the spacer; spacer is at index 0.
        self.layout.insertWidget(len(self._sliders), row)
        self._sliders.append(row)
        self.log_message(f"Added slider '{name}' in range [{minimum}, {maximum}] step {step}.")
        return row  # in case caller wants to keep a handle

    def get_values(self) -> list:
        return [row.value() for row in self._sliders]

    def get_named_values(self) -> dict:
        return {row.name: row.value() for row in self._sliders}

    # ---- UI actions ----
    def on_submit(self):
        vals = self.get_named_values()
        self.log_message(f"Submitted: {vals}")
        print(vals)
        # Emit ordered list of values matching slider order:
        self.valuesSubmitted.emit(self.get_values())

    def log_message(self, msg: str):
        self.log.appendPlainText(msg)



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
                if pad_i.shape is not None:
                    # if it's already been populated, it means its a duplicate in the library
                    # we want to apply these properties to the next pad with the same name
                    continue
                if pad_i.getName() in lines[i - 1]:# or pad_i.getName() in lines[i - 2]:
                    shape_description = lines[i].strip()[1:-2].split("(")[1].split(" ")
                    shape_type = shape_description[0]
                    shape_layer = shape_description[1]
                    # aperture is shape_description[2]

                    if shape_type == "polygon":
                        shape_outline = shape_description[3:]
                        for j in range(len(shape_outline)):
                            shape_outline[j] = float(shape_outline[j])

                    if shape_type == "circle":
                        shape_outline = float(shape_description[2])

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
    component_id_index = 0
    current_component = Component(component_id_index)
    for pad in pads:
        if pad.getID() - last_pad_id >= 16:
            # create new component
            component_id_index += 1
            current_component = Component(component_id_index)
            components.append(current_component)
        
        current_component.addPad(pad)        # add pad to current component
        last_pad_id = pad.getID()

    # create component centre point
    for component in components:
        xsum,ysum = 0,0
        for pad in component.pads:
            x, y = pad.getPosition()
            xsum += x
            ysum += y
        component.setCentre(xsum/len(component.pads), ysum/len(component.pads))

    # update pad centre offsets for each component
    for component in components:
        comp_x,comp_y = component.getPosition()
        for pad in component.pads:
            pad_x, pad_y = pad.getPosition()
            pad.setOffset(comp_x - pad_x, comp_y - pad_y)

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

def makeBoundaryAReasonableFormat(boundary):
    # currently formatted as: [x1 y1 x2 y2 ... xn yn]
    xs = boundary[0::2]
    ys = boundary[1::2]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return ((0,0), (width, height))

def occupancyGridPads(grid_tiles):
    corners = makeBoundaryAReasonableFormat(boundary)
    print(corners)
    number_of_cells_x = corners[1][0] // GRID_SPACING
    number_of_cells_y = corners[1][1] // GRID_SPACING
    print(f"Number of cells: {number_of_cells_x} x {number_of_cells_y}")
    
    for i in range(int(number_of_cells_y)):
        grid_tiles.append([])
        for j in range(int(number_of_cells_x)):
            grid_tiles[i].append(GridTile(j, i))  # 0 means empty cell

    # check if pad overlaps the corner of a grid
    # if the pad.shape is a circle -> treat it as a square with side length of the circles diameter
    # if the pad.shape is a polygon -> find the bounding box of the polygon and treat it as a rectangle
    for pad in pads:
        if pad.shape == "circle":
            print(pad.getName(), pad.ID, pad.getPosition())
            # treat as square
            #diameter = pad.getDiameter()
            diameter = pad.outline
            print(f"diameter: {diameter}")
            x, y = pad.getPosition()
            x1, y1 = x - diameter/2, y - diameter/2
            x2, y2 = x + diameter/2, y + diameter/2
            # find grid cells
            col1 = int(x1 // GRID_SPACING)
            col2 = int(x2 // GRID_SPACING)
            row1 = int(-y1 // GRID_SPACING)
            row2 = int(-y2 // GRID_SPACING)
            print(f"col1: {col1}, col2: {col2}, row1: {row1}, row2: {row2}")
            # mark grid cells as occupied
            for row in range(min(row1, row2), max(row1, row2) + 1):
                for col in range(min(col1, col2), max(col1, col2) + 1):
                    grid_tiles[row][col].addObject(pad)
        elif pad.shape == "polygon":
            # find bounding box
            x_coords, y_coords = zip(*pad.getVertices())
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            # find grid cells
            col1 = int(x1 // GRID_SPACING)
            col2 = int(x2 // GRID_SPACING)
            row1 = int(-y1 // GRID_SPACING)
            row2 = int(-y2 // GRID_SPACING)
            for row in range(min(row1, row2), max(row1, row2) + 1):
                for col in range(min(col1, col2), max(col1, col2) + 1):
                    grid_tiles[row][col].addObject(pad)

    return grid_tiles

def occupancyGridUpdateWireSegment():
    global grid_tiles

    for net in nets:
        for wire_segment in net.wires:
            x1, y1 = wire_segment[0]
            x2, y2 = wire_segment[1]
            col1 = int(x1/1000 // GRID_SPACING)
            col2 = int(x2/1000 // GRID_SPACING)
            row1 = int(-y1/1000 // GRID_SPACING)
            row2 = int(-y2/1000 // GRID_SPACING)

            # Bresenham's line algorithm for grid traversal
            dx = abs(col2 - col1)
            dy = abs(row2 - row1)
            x, y = col1, row1
            sx = 1 if col2 > col1 else -1
            sy = 1 if row2 > row1 else -1

            if dx > dy:
                err = dx / 2.0
                while x != col2:
                    if 0 <= y < len(grid_tiles) and 0 <= x < len(grid_tiles[0]):
                        grid_tiles[y][x].addObject(wire_segment)
                        #net.addOccupancyGridPosition(x, y)
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != row2:
                    if 0 <= y < len(grid_tiles) and 0 <= x < len(grid_tiles[0]):
                        grid_tiles[y][x].addObject(wire_segment)
                        #net.addOccupancyGridPosition(x, y)
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            # Mark the end point
            if 0 <= y < len(grid_tiles) and 0 <= x < len(grid_tiles[0]):
                grid_tiles[y][x].addObject(wire_segment)
                #net.addOccupancyGridPosition(x, y)

def occupancyGridAddVia():
    global grid_tiles

    for net in nets:
        for via in net.vias:
            x, y = via
            col = int(x/1000 // GRID_SPACING)
            row = int(-y/1000 // GRID_SPACING)
            if 0 <= row < len(grid_tiles) and 0 <= col < len(grid_tiles[0]):
                grid_tiles[row][col].addObject(via)

def printGrid():
    for i in range(len(grid_tiles)):
        for j in range(len(grid_tiles[i])):
            tile = grid_tiles[i][j]
            char = '.'
            for obj in tile.objects:
                # Check for via (tuple of length 2)
                if isinstance(obj, tuple) and len(obj) == 2:
                    x, y = obj
                    col = int(x/1000 // GRID_SPACING)
                    row = int(-y/1000 // GRID_SPACING)
                    if col == j and row == i:
                        char = 'X'
                        break  # X takes priority
                # Check for wire segment (tuple of length 3)
                elif isinstance(obj, tuple) and len(obj) == 3:
                    # obj = (start, end, layer)
                    layer = obj[2]
                    if layer == 1:
                        char = '^'
                    elif layer == 2:
                        char = 'v'
                elif isinstance(obj, Pad):
                    char = '@'
            print(char, end=' ')
        print()


def getBlockerType(obj):
    # wires encoded as (start, end, layer)
    if isinstance(obj, tuple) and len(obj) == 3:
        return "wire", int(obj[2])

    if isinstance(obj, Pad):
        return "pad", int(obj.layer)

    return "unknown", None

def reset_astar_weights():
    for row in grid_tiles:
        for tile in row:
            tile.a_star_weight = [None, None]

def aStar2(start, end, start_layer, end_layer, net, nets):
 
    sx = int(start[0] / 1000 // GRID_SPACING)
    sy = int(-start[1] / 1000 // GRID_SPACING)
    ex = int(end[0] / 1000 // GRID_SPACING)
    ey = int(-end[1] / 1000 // GRID_SPACING)
    #print("start,end (cells):", (sx, sy), (ex, ey))

    q = deque()
    # initialise start cell
    #grid_tiles[sy][sx].a_star_weight[start_layer - 1] = 0
    #q.append((sx, sy, start_layer - 1))
    for layer in (0, 1):
        grid_tiles[sy][sx].a_star_weight[layer] = 0
        q.append((sx, sy, layer))

    while q:
        x, y, layer = q.popleft()
        cur_w = grid_tiles[y][x].a_star_weight[layer]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                    (-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or ny >= len(grid_tiles) or nx >= len(grid_tiles[0]):
                continue

            tile = grid_tiles[ny][nx]
            if tile.a_star_weight[layer] is not None:
                continue  # already visited

            # --- blocker check ---
            blocked = False
            for obj in tile.objects:
                obj_type, obj_layer = getBlockerType(obj)

                net_name = None
                if obj_type == "pad":
                    for net_i in nets:
                        if obj in net_i.getPadsInNet():
                            net_name = net_i.name
                            break
                elif obj_type == "wire":
                    for net_i in nets:
                        if obj in net_i.getWiresInNet():
                            net_name = net_i.name
                            break

                if net_name == net.name:
                    blocked = False
                    break

                if (obj_type in ("pad", "wire")) and (obj_layer - 1 == layer):
                    blocked = True
                    break

            if blocked:
                continue

            tile.a_star_weight[layer] = cur_w + 1
            q.append((nx, ny, layer))

        # --- NEW: try switching layer at same (x, y) ---
        other_layer = 1 - layer
        if grid_tiles[y][x].a_star_weight[other_layer] is None:
            # check blockers on the other layer at this same spot
            blocked = False
            for obj in grid_tiles[y][x].objects:
                obj_type, obj_layer = getBlockerType(obj)

                if obj_type in ("pad", "wire") and obj_layer - 1 == other_layer:
                    blocked = True
                    break

            if not blocked:
                grid_tiles[y][x].a_star_weight[other_layer] = cur_w + 1
                q.append((x, y, other_layer))

    #print("A* fill completing... Solving...")
    #print(f"End cell weights: Top: {grid_tiles[ey][ex].a_star_weight[0]}, Bottom: {grid_tiles[ey][ex].a_star_weight[1]}")
    # start at ex, ey
    # find adjacent cells' values
    # move new cell is the adjacent cell with the lowest weight
    # repeat until 0
    
    # normalize layers to 0 (top) or 1 (bottom)
    start_layer = 0 if start_layer == 1 else 1
    end_layer   = 0 if end_layer == 1 else 1

    current_tile = [ex, ey, end_layer]  # (x, y, layer)
    solve_path = []

    while True:
        x, y, layer = current_tile
        solve_path.append(current_tile)

        current_weight = grid_tiles[y][x].a_star_weight[layer]
        if current_weight is None:
            print("No path (hit None).")
            break
        if current_weight == 0 and [x, y, layer] == [sx, sy, start_layer]:
            break

        # find adjacent candidates
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid_tiles[0]) and 0 <= ny < len(grid_tiles):
                w = grid_tiles[ny][nx].a_star_weight[layer]
                if w is not None and w >= 0 and w < current_weight:
                    adjacent.append([nx, ny, layer, w])

        # check via (stay in place, switch layers)
        other_layer = 1 - layer
        ow = grid_tiles[y][x].a_star_weight[other_layer]
        if ow is not None and ow >= 0 and ow < current_weight:
            adjacent.append([x, y, other_layer, ow])

        if not adjacent:
            print(f"Stuck: no lower-weight neighbors at {current_tile}, traveled a distance of {len(solve_path)} with {int(np.sqrt((ex - x) ** 2 + (ey - y) ** 2))} to go.")
            break

        # pick best candidate
        current_tile = min(adjacent, key=lambda pos: pos[3])[:3]

    #print("A* path found:", solve_path[::-1])  # flip to start→end
    return solve_path[::-1]  # return the path from start to end


def printGrid3():
    print("Top layer:")
    for i in range(len(grid_tiles)):
        for j in range(len(grid_tiles[i])):
            w = grid_tiles[i][j].a_star_weight[0]
            print(f"{w:02d}" if w is not None else "@@", end=" ")
        print("")
    print("")

    print("Bottom layer:")
    for i in range(len(grid_tiles)):
        for j in range(len(grid_tiles[i])):
            w = grid_tiles[i][j].a_star_weight[1]  # fixed check
            print(f"{w:02d}" if w is not None else "@@", end=" ")
        print("")



def demo_build(panel: SliderPanel):
    """
    Example: build a few differently ranged sliders with 0.1 resolution.
    Replace with your own creation code.
    """
    panel.add_slider("Route Clearance",       0.0,  5.0,  2.0,  step=0.1)
    panel.add_slider("Offset",   -10.0,  10.0,  0.0,  step=0.1)
    panel.add_slider("Throttle",   0.0,   1.0,  0.5,  step=0.1)
    panel.add_slider("Cutoff Hz",  5.0, 200.0, 50.0,  step=0.1)  # still 0.1 resolution

def approximate_gradient(component, pos, nets, delta=1e-6):
    x, y = pos
    # Partial derivative wrt x
    f_x1 = total_length(component, (x + delta, y), nets)
    f_x2 = total_length(component, (x - delta, y), nets)
    grad_x = (f_x1 - f_x2) / (2 * delta)
    
    
    # Partial derivative wrt y
    f_y1 = total_length(component, (x, y + delta), nets)
    f_y2 = total_length(component, (x, y - delta), nets)
    grad_y = (f_y1 - f_y2) / (2 * delta)
    
    return (grad_x, grad_y)

def gradient_descent_move(component, nets_to_minimise, learning_rate=0.1, max_iter=500, tolerance=1e-6):
    pos = component.getPosition()
    for i in range(max_iter):
        grad = approximate_gradient(component, pos, nets_to_minimise)
        grad_magnitude = (grad[0]**2 + grad[1]**2)**0.5
        if grad_magnitude < tolerance:
            print("grad no good", grad)
            break  # gradient is too small, stop
        
        new_pos = (pos[0] - learning_rate * grad[0], pos[1] - learning_rate * grad[1])
        
        current_length = total_length(component, pos, nets_to_minimise)
        new_length = total_length(component, new_pos, nets_to_minimise)
        
        if new_length >= current_length:
            # No improvement, can try reducing learning rate or stop
            print('nah fam its coooked')
            break
        diff_x, diff_y = new_pos[0]-pos[0], new_pos[1]-pos[1]
        component.move(new_pos[0], new_pos[1])
        for comp in component_cluster:
            x,y = comp.getPosition()
            comp.move(x+diff_x, y+diff_y)
        pos = new_pos
        # print(f"Iteration {i+1}: Position={pos}, Connection Length={new_length}")
    
    return pos

def total_length(component, pos, list_nets):
    orginal_pos = component.getPosition()
    component.move(pos[0], pos[1])
    sum = 0
    for net in list_nets:
        sum += net.getLength()
    component.move(orginal_pos[0], orginal_pos[1])
    return sum


def place(component):
    nets_to_minimise = []
    for pad in component.pads: # only for components with 2 pads
        for net in nets:
            net_pads = net.getPadsInNet()
            if pad in net_pads and net not in nets_to_minimise: # find the net of the pad
                nets_to_minimise.append(net)
    gradient_descent_move(component, nets_to_minimise)

def all_net_length():
    sum = 0
    for net in nets:
        sum += net.getLength()
    return sum

def simulated_annealing(initial_temp=1000, final_temp=1, alpha=0.9, max_iter=1000, step_size=1.0):
    """
    components: list of components with .getPosition() and .move(x, y)
    total_length_func: function returning total MST trace length of all nets
    initial_temp: start temperature
    final_temp: end temperature
    alpha: cooling rate (0 < alpha < 1)
    max_iter: max iterations per temperature
    step_size: max distance to move component randomly per step
    """
    
    # Initialize positions
    positions = {comp: comp.getPosition() for comp in components}
    current_length = all_net_length()
    temp = initial_temp
    
    while temp > final_temp:
        for _ in range(max_iter):
            # Randomly pick a component and propose a small move
            comp = random.choice(components)
            old_pos = positions[comp]
            
            # Propose new position (small random shift)
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            
            # Move component to new position
            comp.move(new_pos[0], new_pos[1])
            
            # Calculate new total length
            new_length = all_net_length()
            
            # Calculate change in "energy" (cost)
            delta = new_length - current_length
            
            # Decide whether to accept new position
            if delta < 0:
                # Improved placement: accept
                positions[comp] = new_pos
                current_length = new_length
            else:
                # Accept worse solution with a probability
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    positions[comp] = new_pos
                    current_length = new_length
                else:
                    # Revert move
                    comp.move(old_pos[0], old_pos[1])
                    
        # Cool down temperature
        temp *= alpha
        
        # Optionally print progress
        print(f"Temperature: {temp:.2f}, Current total length: {current_length:.2f}")
        
    return positions

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def unit_vector(p1, p2):
    dist = distance(p1, p2)
    if dist == 0:
        return (0, 0)
    return ((p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist)

def force_directed_placement(iterations=1000, k_attract=0.1, k_repel=1000, max_disp=1.0):
    """
    components: list of components with .getPosition() returning (x, y), and .move(x, y)
    nets: list of nets, each net is a list of components connected
    k_attract: constant for attractive force strength
    k_repel: constant for repulsive force strength
    max_disp: maximum displacement per iteration to avoid jittering
    """

    for _ in range(iterations):
        # Initialize displacement dict for each component
        disp = {comp: [0.0, 0.0] for comp in components}

        # Calculate repulsive forces (between all pairs)
        for i, c1 in enumerate(components):
            p1 = c1.getPosition()
            for j in range(i + 1, len(components)):
                c2 = components[j]
                p2 = c2.getPosition()
                delta = unit_vector(p2, p1)  # repel direction from c2 to c1
                dist = distance(p1, p2)
                if dist == 0:
                    dist = 0.01  # avoid division by zero
                
                force = k_repel / (dist * dist)  # inverse square repulsion
                disp[c1][0] += delta[0] * force
                disp[c1][1] += delta[1] * force
                disp[c2][0] -= delta[0] * force
                disp[c2][1] -= delta[1] * force

        # Calculate attractive forces (along nets)
        for net in nets:
            net_comps = net.getComponents()  # Assume net can provide connected components
            for i, c1 in enumerate(net_comps):
                p1 = c1.getPosition()
                for j in range(i + 1, len(net_comps)):
                    c2 = net_comps[j]
                    p2 = c2.getPosition()

                    delta = unit_vector(p1, p2)  # attract direction from c1 to c2
                    dist = distance(p1, p2)

                    force = k_attract * (dist * dist)  # proportional to square distance attract
                    disp[c1][0] += delta[0] * force
                    disp[c1][1] += delta[1] * force
                    disp[c2][0] -= delta[0] * force
                    disp[c2][1] -= delta[1] * force

        # Move components by displacement vector clipped to max_disp
        for comp in components:
            dx, dy = disp[comp]
            disp_len = math.hypot(dx, dy)
            if disp_len > max_disp:
                dx = dx / disp_len * max_disp
                dy = dy / disp_len * max_disp

            old_x, old_y = comp.getPosition()
            new_x, new_y = old_x + dx, old_y + dy
            comp.move(new_x, new_y)

def gradient_descent(iter=1):
    for i in range(iter):
        component_cluster = []
        for component in components:
            place(component)
            component_cluster.append(component)

def count_intersections():
    """
    Count the number of intersections between line segments.
    Each segment is defined as ((x1, y1), (x2, y2)).
    """
    segments = []
    for net in nets:  # get all vectors in netlist
        for i in range(len(net.pads) - 1):
            segments.append((net.pads[i].getPosition(), net.pads[i+1].getPosition()))
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """Check if point q lies on segment pr"""
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def segments_intersect(s1, s2):
        """Check if segments s1 and s2 intersect"""
        p1, q1 = s1
        p2, q2 = s2

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases: collinear points
        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True

        return False

    n = len(segments)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if segments_intersect(segments[i], segments[j]):
                count += 1
    return count

def optimise_rotation():
    rotates = [0, 90, 180, 270]
    if len(components) < 10:  # O(n!) Joel would be ashamed :(
        min_intersects = math.inf
        best_rotations = {}
        
        # Store original positions to restore later
        original_positions = {}
        for component in components:
            original_positions[component] = [pad.getPosition() for pad in component.pads]
        
        # Generate all possible rotation combinations
        from itertools import product
        
        for rotation_combo in product(rotates, repeat=len(components)):
            # Apply rotations to all components
            for i, component in enumerate(components):
                # Reset to original position first
                for j, pad in enumerate(component.pads):
                    pad.setPosition(*original_positions[component][j])
                # Apply rotation
                component.rotate(rotation_combo[i])
            
            # Calculate intersections for this combination
            
            
            count = count_intersections()
            
            if count < min_intersects:
                min_intersects = count
                best_rotations = {component: rotation_combo[i] for i, component in enumerate(components)}
                
                # If we found zero intersections, we can stop
                if count == 0:
                    print("Found zero intersections! Stopping early.")
                    break
        
        # Apply the best rotation combination
        for component in components:
            # Reset to original position first
            for j, pad in enumerate(component.pads):
                pad.setPosition(*original_positions[component][j])
            # Apply best rotation
            best_theta = best_rotations[component]
            component.rotate(best_theta)
            component.setTheta(best_theta)
        
        print(f"Final result: {min_intersects} intersections")
    else:
        print(f"Too many components ({len(components)}) for exhaustive search")

def visualise(packer, area, rectangles, buffer, minimal_width, minimal_height):
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Draw the bin boundary
    bin_width, bin_height = area
    bin_rect = patches.Rectangle((0, 0), bin_width, bin_height, 
                                linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_patch(bin_rect)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Draw each packed rectangle
    for i, rect in enumerate(packer[0]):
        color = colors[i % len(colors)]
        rectangle = patches.Rectangle((rect.x+(buffer-1)*rect.width/2, rect.y+(buffer-1)*rect.height/2), rect.width/buffer, rect.height/buffer,
                                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rectangle)
        
        # Add text label with rectangle info
        ax.text(rect.x + rect.width/2, rect.y + rect.height/2, 
                f'{rect.width}x{rect.height}', 
                ha='center', va='center', fontsize=8, fontweight='bold')

    # Set up the plot
    ax.set_xlim(0, bin_width*1.3)
    ax.set_ylim(0, bin_height*1.3)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title(f'PCB layout optimisation\nBoard size: {bin_width}x{bin_height}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    
    
    minimal_rect = patches.Rectangle((0, 0), minimal_width*(1+(buffer-1)/2), minimal_height*(1+(buffer-1)/2),
                                    linewidth=2, edgecolor='gray', facecolor='none', 
                                    linestyle='--', alpha=0.8)
    ax.add_patch(minimal_rect)

    # Print packing efficiency
    total_rect_area = sum(w * h for w, h in rectangles)
    bin_area = bin_width * bin_height
    efficiency = (total_rect_area / bin_area) * 100
    
    print(f"\nPacking efficiency: {efficiency:.1f}% ({total_rect_area}/{bin_area})")
    print(f"Minimal outside dimensions: {minimal_width:.2f} x {minimal_height:.2f}")

    plt.tight_layout()
    
    plt.show()


    

def optimise_board(area, buffer=1, max_iterations=1, tolerance=0.1):
    """
    Iteratively optimize board size by using minimal dimensions from previous iteration
    as new board size until no further improvement is possible.
    """
    footprints = []
    for comp in components:
        comp.setDimensions()
        footprints.append((round(buffer*comp.dimensions[0],2), round(buffer*comp.dimensions[1], 2)))

    num_components = len(footprints)
    current_area = area
    iteration = 0
    final_packer = None
    min_width = False
    min_height = False
    CONVERGENCE_FACTOR = 0.95
    old_area = area
    
    print(f"Starting optimization with initial board size: {current_area}")
    
    while iteration < max_iterations:
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Trying board size: {current_area}")
        
        # Try packing with current area
        packer = newPacker()
        for r in footprints:
            packer.add_rect(*r)
        packer.add_bin(current_area[0], current_area[1])
        packer.pack()

        # Check if packing was successful
        print(num_components, len(packer[0]))
        if not packer[0] or len(packer[0]) < num_components:  # No rectangles were packed
            print("Packing failed! Reverting to previous size.")
            current_area = old_area
            if not min_width:
                min_width = True
            elif not min_height:
                min_height = True
            continue
        
        # Calculate minimal outside dimensions from packed rectangles
        minimal_width = max(rect.x + rect.width for rect in packer[0]) if packer[0] else 0
        minimal_height = max(rect.y + rect.height for rect in packer[0]) if packer[0] else 0
        
        print(f"Packed successfully! Minimal dimensions: {minimal_width:.2f} x {minimal_height:.2f}")
        
        # Check if we made significant improvement
        width_improvement = current_area[0] - minimal_width
        height_improvement = current_area[1] - minimal_height
        
        old_area = current_area
        if width_improvement < tolerance and height_improvement < tolerance:
            print(f"Converged! Improvement too small: {width_improvement:.2f} x {height_improvement:.2f}")
            print(min_width, min_height)
            # if not min_width:
            #     current_area = (CONVERGENCE_FACTOR* minimal_width, minimal_height)
            # elif not min_height:
            #     current_area = (minimal_width, CONVERGENCE_FACTOR* minimal_height)
            # else:
            #     print(num_components, len(packer[0]))
            break
        else:
            current_area = (minimal_width, minimal_height)
        # Store current successful packing for visualization
        final_packer = packer
        final_area = current_area
        final_minimal_width = minimal_width
        final_minimal_height = minimal_height
        
        
        iteration += 1

    # Update component positions for this iteration
    used = []
    for comp in components:
        for i, r in enumerate(final_packer[0]):
            if (buffer*comp.dimensions[0] == r.width and buffer*comp.dimensions[1] == r.height and i not in used):
                comp.move(r.x + r.width/2, r.y + r.height/2)
                used.append(i)
                break
            elif (buffer*comp.dimensions[0] == r.height and buffer*comp.dimensions[1] == r.width and i not in used):
                comp.rotate(90)
                comp.move(r.x + r.width/2, r.y + r.height/2)
                used.append(i)
                break

    print(f"\nOptimization complete after {iteration} iterations")
    print(f"Final board size: {final_area}")
    print(f"Final minimal dimensions: {final_minimal_width:.2f} x {final_minimal_height:.2f}")
    
    # Visualize the final result
    # visualise(final_packer, area, footprints, buffer, final_minimal_width, final_minimal_height)



# ---- Worker that does the routing AFTER the button press ----
class Worker(QObject):
    progress = pyqtSignal(str)   # text lines to append to GUI log
    finished = pyqtSignal(str)   # final message

    def __init__(self, slider_values):
        super().__init__()
        self.slider_values = slider_values  # use if your routing needs them

    def run(self):
        try:
            # 1) Read the DSN file
            self.progress.emit("Reading .DSN file...")
            # If you need to clear old state, do it here.
            # e.g., nets.clear() or similar, depending on your codebase

            # Make sure grid_tiles is available
            global grid_tiles
            grid_tiles = []  # reset / new grid

            processDSNfile(f"DSN/{FILE_NAME}.dsn")
            self.progress.emit("Successfully read .DSN file")

            # 2) Build occupancy grid
            self.progress.emit("Building occupancy grid for pads...")
            occupancyGridPads(grid_tiles)

            self.progress.emit("Updating occupancy grid for existing wire segments...")
            occupancyGridUpdateWireSegment()

            # Optional: show grid in console (redirected to GUI if desired)
            # If printGrid() prints to stdout, you can keep it; here we just note it:
            self.progress.emit("Initial grid ready.")
            # printGrid()

            # 3) Route each net
            total_nets = len(nets)
            for net_idx, net in enumerate(nets, start=1):
                pads = net.getPadsInNet()
                if not pads or len(pads) < 2:
                    self.progress.emit(f"Net '{net.name}': not enough pads to route.")
                    continue

                for i in range(len(pads) - 1):
                    pad = pads[i]
                    next_pad = pads[i + 1]

                    # multiple both elements of pad tuple by 1000
                    pad_mils = (int(pad.getPosition()[0] * 1000), int(pad.getPosition()[1] * 1000))
                    next_pad_mils = (int(next_pad.getPosition()[0] * 1000), int(next_pad.getPosition()[1] * 1000))

                    reset_astar_weights()
                    path = aStar2(pad_mils, next_pad_mils, 1, 1, net, nets)

                    # make path a wire
                    for j in range(len(path) - 1):
                        start = path[j][:2]
                        end = path[j + 1][:2]
                        start = (start[0] * GRID_SPACING * 1000, -start[1] * GRID_SPACING * 1000)  # convert to mils
                        end = (end[0] * GRID_SPACING * 1000, -end[1] * GRID_SPACING * 1000)      # convert to mils
                        net.addWireSegment(start, end, 1)

                    occupancyGridUpdateWireSegment()

                    # Write SES after each connection (as per your code)
                    processSESfile(f"SES/{FILE_NAME}.ses")

                    # This was your print; now goes to log:
                    self.progress.emit(
                        f"{i+1} / {len(pads)-1} connections done for net '{net.name}' "
                        f"({net_idx}/{total_nets} nets)"
                    )

            self.finished.emit("Routing complete. SES updated.")
        except Exception as e:
            # Bubble up errors to the log as well
            self.finished.emit(f"Routing aborted with error: {e!r}")

# ---- Main wiring: start routing when Submit is clicked ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SliderPanel()
    demo_build(w)
    w.show()

    # Start worker when the Submit button is pressed
    thread_holder = {"thread": None, "worker": None}  # keep references alive

    def start_routing(slider_values_list):
        # Optional: map slider list to named dict if you need names
        named = w.get_named_values()
        w.log_message(f"Starting routing with sliders: {named}")

        # Guard against restarting while a job is running
        if thread_holder["thread"] is not None:
            w.log_message("A routing job is already running.")
            return

        thread = QThread()
        worker = Worker(slider_values_list)
        worker.moveToThread(thread)

        # Wiring
        thread.started.connect(worker.run)
        worker.progress.connect(w.log_message)
        worker.finished.connect(w.log_message)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Keep refs so they don’t get GC’d
        thread_holder["thread"] = thread
        thread_holder["worker"] = worker

        def cleanup():
            thread_holder["thread"] = None
            thread_holder["worker"] = None
        thread.finished.connect(cleanup)

        thread.start()

    # Hook the GUI button signal to start routing
    w.valuesSubmitted.connect(start_routing)

    sys.exit(app.exec_())
