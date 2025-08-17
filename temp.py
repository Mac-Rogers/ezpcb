import numpy as np
import math
import random
from collections import deque
from rectpack import newPacker
import matplotlib.pyplot as plt
import matplotlib.patches as patches


pads = []
nets = []
components = []

GRID_SPACING = 1.5
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
        self.a_star_weight = [] # 2 element array for top and bottom layer
    
    def addObject(self, object):
        self.objects.append(object)
    
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
            row1 = int(y1 // GRID_SPACING)
            row2 = int(y2 // GRID_SPACING)
            # mark grid cells as occupied
            for row in range(row1, row2 + 1):
                for col in range(col1, col2 + 1):
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



def aStar(start, end, nets):
    """
    start,end in mils (x,y). Fills BOTH layers' weights:
      grid_tiles[y][x].a_star_weight[1] -> Top
      grid_tiles[y][x].a_star_weight[2] -> Bottom
    Blocks:
      - Vias from nets: both layers
      - Wires from nets: their own layer
      - Pads already attached to tiles: their own layer
    Diagonals allowed. Start = 1 on both layers (even if blocked).
    """
    # --- mils -> grid cells (your convention) ---
    sx = int(start[0] / 1000 // GRID_SPACING)
    sy = int(-start[1] / 1000 // GRID_SPACING)
    ex = int(end[0]   / 1000 // GRID_SPACING)
    ey = int(-end[1]  / 1000 // GRID_SPACING)
    print("start,end (cells):", (sx, sy), (ex, ey))

    if not grid_tiles or not grid_tiles[0]:
        return

    H = len(grid_tiles)
    W = len(grid_tiles[0])

    def in_bounds(x, y): return 0 <= x < W and 0 <= y < H

    # init weights (index 0 unused; 1=Top, 2=Bottom)
    for y in range(H):
        for x in range(W):
            grid_tiles[y][x].a_star_weight = [None, None, None]

    # ---- build blocked sets from nets (wires & vias) ----
    blocked = {1: set(), 2: set()}  # per-layer sets of (x,y)

    def to_cell(pt):
        return (int(pt[0] / 1000 // GRID_SPACING),
                int(-pt[1] / 1000 // GRID_SPACING))

    def bresenham_cells(c1, c2):
        x1, y1 = c1; x2, y2 = c2
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        sx = 1 if x2 >= x1 else -1
        sy = 1 if y2 >= y1 else -1
        x, y = x1, y1
        cells = []
        if dx >= dy:
            err = dx // 2
            while x != x2:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            cells.append((x2, y2))
        else:
            err = dy // 2
            while y != y2:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            cells.append((x2, y2))
        return cells

    # wires
    for net in nets:
        for (p1, p2, layer_id) in net.wires:
            layer_id = 1 if layer_id == 1 else 2  # normalize
            c1 = to_cell(p1); c2 = to_cell(p2)
            for (cx, cy) in bresenham_cells(c1, c2):
                if in_bounds(cx, cy):
                    blocked[layer_id].add((cx, cy))
        # vias: assume net.vias are mils (x,y); block both layers
        for pos in net.vias:
            cx, cy = to_cell(pos)
            if in_bounds(cx, cy):
                blocked[1].add((cx, cy))
                blocked[2].add((cx, cy))

    # pads already on tiles: block their own layer
    def obj_kind(o):
        k = getattr(o, "kind", None)
        if k: return str(k).lower()
        name = o.__class__.__name__.lower()
        if "via" in name: return "via"
        if "pad" in name: return "pad"
        if "wire" in name or "net" in name: return "wire"
        return "unknown"

    def obj_layer_id(o):
        lay = getattr(o, "layer", None)
        if lay is None: return None
        if isinstance(lay, int): return 1 if lay == 1 else (2 if lay == 2 else None)
        s = str(lay).lower()
        if s.startswith("t") or s == "1": return 1
        if s.startswith("b") or s == "2": return 2
        return None

    for y in range(H):
        for x in range(W):
            for o in grid_tiles[y][x].objects:
                k = obj_kind(o)
                if k == "via":
                    blocked[1].add((x, y)); blocked[2].add((x, y))
                elif k == "pad":
                    lid = obj_layer_id(o)
                    if lid in (1, 2): blocked[lid].add((x, y))
                elif k == "wire":
                    lid = obj_layer_id(o)
                    if lid in (1, 2): blocked[lid].add((x, y))

    def is_blocked(x, y, lid): return (x, y) in blocked[lid]

    # ---- BFS fill per layer (diagonals allowed) ----
    nbrs = [(-1,-1),(0,-1),(1,-1),
            (-1, 0),        (1, 0),
            (-1, 1),(0, 1),(1, 1)]

    def bfs_layer(layer_id):
        if not in_bounds(sx, sy): return
        grid_tiles[sy][sx].a_star_weight[layer_id] = 1  # seed even if blocked
        q = deque([(sx, sy)])
        while q:
            x, y = q.popleft()
            w = grid_tiles[y][x].a_star_weight[layer_id]
            for dx, dy in nbrs:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny): continue
                if grid_tiles[ny][nx].a_star_weight[layer_id] is not None: continue
                if is_blocked(nx, ny, layer_id): continue
                grid_tiles[ny][nx].a_star_weight[layer_id] = w + 1
                q.append((nx, ny))

    bfs_layer(1)  # Top
    bfs_layer(2)  # Bottom


def printGrid2():
    """Print Top (1) and Bottom (2) layers once."""
    if not grid_tiles or not grid_tiles[0]:
        print("(empty grid)"); return
    H = len(grid_tiles); W = len(grid_tiles[0])

    # reuse block check for display (derive from tile.objects only; wires from nets
    # are already accounted for by aStar into the weights; showing blocks as '##'
    # based on objects makes pads/vias obvious; cells blocked by net wires will show
    # as '..' unless you also attach wire-objects to tiles)
    def obj_kind(o):
        k = getattr(o, "kind", None)
        if k: return str(k).lower()
        name = o.__class__.__name__.lower()
        if "via" in name: return "via"
        if "pad" in name: return "pad"
        if "wire" in name or "net" in name: return "wire"
        return "unknown"
    def obj_layer_id(o):
        lay = getattr(o, "layer", None)
        if lay is None: return None
        if isinstance(lay, int): return 1 if lay == 1 else (2 if lay == 2 else None)
        s = str(lay).lower()
        if s.startswith("t") or s == "1": return 1
        if s.startswith("b") or s == "2": return 2
        return None
    def is_blocked_by_objects(x, y, lid):
        for o in grid_tiles[y][x].objects:
            k = obj_kind(o)
            if k == "via": return True
            if k in ("pad", "wire") and obj_layer_id(o) == lid:
                return True
        return False

    def dump(lid, title):
        print(f"\n=== {title} (layer {lid}) ===")
        for y in (range(H)):
            row = []
            for x in range(W):
                if is_blocked_by_objects(x, y, lid):
                    row.append("##")
                else:
                    ws = grid_tiles[y][x].a_star_weight
                    w = ws[lid] if ws and len(ws) > lid else None
                    row.append(".." if w is None else f"{w:02d}")
            print(" ".join(row))

    dump(1, "Top")
    dump(2, "Bottom")
    """
    Print Top (1) and Bottom (2) layers. 
      - blocked -> "##"
      - None (unvisited) -> ".."
      - else -> 2-digit weight
    """
    if not grid_tiles or not grid_tiles[0]:
        print("(empty grid)")
        return

    H = len(grid_tiles)
    W = len(grid_tiles[0])

    def obj_kind(o):
        k = getattr(o, "kind", None)
        if k: return str(k).lower()
        cname = o.__class__.__name__.lower()
        if "via" in cname: return "via"
        if "pad" in cname: return "pad"
        if "wire" in cname or "net" in cname: return "wire"
        return "unknown"

    def obj_layer_id(o):
        lay = getattr(o, "layer", None)
        if lay is None:
            return None
        if isinstance(lay, int):
            return 1 if lay == 1 else (2 if lay == 2 else None)
        s = str(lay).lower()
        if s.startswith("t"): return 1
        if s.startswith("b"): return 2
        if s == "1": return 1
        if s == "2": return 2
        return None

    def is_blocked(x, y, layer_id):
        for o in grid_tiles[y][x].objects:
            k = obj_kind(o)
            if k == "via":
                return True
            if k in ("pad", "wire") and obj_layer_id(o) == layer_id:
                return True
        return False

    def dump_layer(layer_id, title):
        print(f"\n=== {title} (layer {layer_id}) ===")
        for y in (range(H)):  # top row first
            row_str = []
            for x in range(W):
                if is_blocked(x, y, layer_id):
                    row_str.append("##")
                else:
                    w = None
                    ws = grid_tiles[y][x].a_star_weight
                    if ws and len(ws) > layer_id:
                        w = ws[layer_id]
                    row_str.append(".." if w is None else f"{w:02d}")
            print(" ".join(row_str))

    dump_layer(1, "Top")
    dump_layer(2, "Bottom")


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
def optimise_board(area, buffer=1, max_iterations=20, tolerance=0.1):
    """
    Iteratively optimize board size by shrinking width first, then height, until packing fails or no significant improvement.
    """
    footprints = []
    for comp in components:
        comp.setDimensions()
        footprints.append((round(buffer * comp.dimensions[0], 2), round(buffer * comp.dimensions[1], 2)))

    num_components = len(footprints)
    current_area = area
    iteration = 0
    final_packer = None
    phase = "width"  # Start by shrinking width
    best_area = area
    best_width, best_height = area
    converged = False

    print(f"Starting optimization with initial board size: {current_area}")

    while iteration < max_iterations and not converged:
        print(f"\n--- Iteration {iteration + 1} ({phase}-minimization) ---")
        print(f"Trying board size: {current_area}")

        # Try packing with current area
        packer = newPacker()
        for r in footprints:
            packer.add_rect(*r)
        packer.add_bin(current_area[0], current_area[1])
        packer.pack()

        # Check if packing was successful
        if not packer[0] or len(packer[0]) < num_components:
            
            print("Packing failed! Reverting to last best size and switching dimension.")
            # Switch phase when failed
            if phase == "width":
                current_area = (best_width, best_height)
                phase = "height"
                # Reset iteration count for height shrink phase if desired
            elif phase == "height":
                converged = True
            iteration += 1
            continue

        # Calculate minimal packed dimensions
        minimal_width = max(rect.x + rect.width for rect in packer[0])
        minimal_height = max(rect.y + rect.height for rect in packer[0])

        print(f"Packed successfully! Minimal dimensions: {minimal_width:.2f} x {minimal_height:.2f}")

        # Set final_packer to the first successful packer if not set yet
        if final_packer is None:
            final_packer = packer
            best_width, best_height = minimal_width, minimal_height
            best_area = (best_width, best_height)

        # Update the best result if improved
        if minimal_width * minimal_height < best_width * best_height:
            best_width, best_height = minimal_width, minimal_height
            best_area = (best_width, best_height)
            final_packer = packer

        # Check convergence
        width_improvement = current_area[0] - minimal_width
        height_improvement = current_area[1] - minimal_height

        # Decide on next size or phase
        if phase == "width":
            if width_improvement < tolerance:
                print(f"Width improvement too small: {width_improvement:.2f}. Switching to height minimization.")
                phase = "height"
            else:
                current_area = (minimal_width * 0.95, current_area[1])
        elif phase == "height":
            if height_improvement < tolerance:
                print(f"Height improvement too small: {height_improvement:.2f}. Converged.")
                converged = True
            else:
                current_area = (current_area[0], minimal_height * 0.95)

        iteration += 1

    # Check if we have a valid final_packer
    if final_packer is None:
        print("ERROR: No successful packing found! Cannot proceed.")
        return

    # Update component positions for final packing

    for comp in components:
        print(f"Component began at {round(comp.getPosition()[0], 2), round(comp.getPosition()[1], 2)} with dimensions {comp.dimensions}")
    
    used = []
    for comp in components:
        for i, r in enumerate(final_packer[0]):
            if (buffer * round(comp.dimensions[0], 2) == round(r.width, 2) and
                buffer * round(comp.dimensions[1], 2) == round(r.height, 2) and
                i not in used):
                comp.move(r.x + r.width / 2, r.y + r.height / 2)
                used.append(i)
                break
            elif (buffer * round(comp.dimensions[1], 2) == round(r.height, 2) and
                  buffer * round(comp.dimensions[0], 2) == round(r.width, 2) and
                  i not in used):
                comp.rotate(90)
                comp.move(r.x + r.width / 2, r.y + r.height / 2)
                used.append(i)
                break
    for comp in components:
        print(f"Component placed at {comp.getPosition()} with dimensions {comp.dimensions}")
    print(f"\nOptimization complete after {iteration} iterations")
    print(f"Final board size: {best_area}")
    print(f"Final minimal dimensions: {best_width:.2f} x {best_height:.2f}")

    # Visualize the final result
    visualise(final_packer, area, footprints, buffer, best_width, best_height)



grid_tiles = []
processDSNfile(f"DSN/{FILE_NAME}.dsn")



# components[1].rotate(180)
optimise_board((50, 10))  # Start with a reasonable initial size


# optimise_rotation()
# simulated_annealing()
# force_directed_placement()
# gradient_descent()


# occupancyGridPads(grid_tiles)

# nets[1].addWireSegment((20*1000,0), (20*1000,-20*1000), 1)
#nets[2].addWireSegment((0,0), (30*1000,-10*1000), 2)
#nets[2].addVia((20*1000,-15*1000))
# occupancyGridUpdateWireSegment()

# printGrid()
veryBasicRoute()
# aStar((0, 0), (70*1000, -20*1000), nets)
# printGrid2()



processSESfile(f"SES/{FILE_NAME}.ses")
