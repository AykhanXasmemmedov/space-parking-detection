import numpy as np

def annotation_txt_read(path,h,w):
    
    box_coordinate=[]
    
    with open(path) as f:
        lines = f.readlines()
        
        for i in range(len(lines)):
            coordinates=lines[i].split(' ')
            
            coordinates=np.array([float(coordinates[1]),float(coordinates[2]),\
                                  float(coordinates[3]),float(coordinates[4][:-1])])
            
            (box_cen_x,box_cen_y,box_width,box_height)=(coordinates*np.array([w,h,w,h])).astype('int')

            x0=int(box_cen_x-(box_width/2))
            y0=int(box_cen_y-(box_height/2))

            x1=int(x0+box_width)
            y1=int(y0+box_height)
            
            box_coordinate.append([x0,y0,x1,y1])
            
    return box_coordinate
            

   
