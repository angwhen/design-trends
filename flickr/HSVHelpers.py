import math

#https://stackoverflow.com/questions/35113979/calculate-distance-between-colors-in-hsv-space
def project_hsv_to_hsv_cone(p): #input out of 255
    return (p[1]/255*p[2]/255*math.sin(p[0]/255*2*math.pi), p[1]/255*p[2]/255*math.cos(p[0]/255*2*math.pi), p[2]/255)

def hsv_cone_coords_to_hsv(p): #returns out of 255
    val = p[2]
    if val != 0:
        sat = math.sqrt((p[0]*p[0]+p[1]*p[1])/(val*val))
    else:
        sat = 0
    if sat!=0 and val != 0:
        hue1 = math.asin(p[0]/sat/val) #figure out which result based on cosine
        hue2 = math.acos(p[1]/sat/val)
        if hue1 == hue2:
            hue = hue1
        elif hue1 < 0:
            if hue2 <= math.pi/2:
                hue = 2*math.pi+hue1
            else:
                hue = math.pi-hue1
        else:
            hue = hue2
    else:
        hue = 0
    return (hue*255/(2*math.pi),sat*255,val*255)

def get_hsv_color_from_hex(hex_color):
    return [0,0,0]
