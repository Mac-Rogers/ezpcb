import numpy as np
import math
from collections import deque

pads = []
nets = []
components = []

GRID_SPACING = 1.5

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

    def move(self, new_x, new_y):
        current_x, current_y = self.position
        diff_x, diff_y = new_x - current_x, new_y - current_y
        self.position = (new_x, new_y)
        for pad in self.pads:
            pad_x, pad_y = pad.getPosition()
            pad.setPosition(diff_x + pad_x, diff_y + pad_y)


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



def aStar(start, end, nets, start_layer, end_layer):
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

    # solve the maze
    print(ex, ey, grid_tiles[ey][ex].a_star_weight)

    

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
    


grid_tiles = []
processDSNfile("DSN/basic1layerRoute.dsn")

vector_list = []
for pad in components[0].pads: # only for components with 2 pads
    padx, pady = pad.getPosition()
    for net in nets:
        net_pads = net.getPadsInNet()
        if pad in net_pads: # find the net of the pad
            pass
           

occupancyGridPads(grid_tiles)

nets[1].addWireSegment((20*1000,0), (20*1000,-20*1000), 1)
#nets[2].addWireSegment((0,0), (30*1000,-10*1000), 2)
#nets[2].addVia((20*1000,-15*1000))
occupancyGridUpdateWireSegment()

printGrid()
#veryBasicRoute()
print("a star time")
aStar((5*1000, 10*1000), (70*1000, -20*1000), nets, 1, 2)
printGrid2()



processSESfile("SES/basic1layerRoute.ses")
