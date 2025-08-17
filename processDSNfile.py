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
FILE_NAME = "test2"

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

            processDSNfile("DSN/basic1layerCrossover.dsn")
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
                    processSESfile("SES/basic1layerCrossover.ses")

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
