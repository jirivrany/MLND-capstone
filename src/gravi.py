import numpy as np
Kappa = 6.67e-11


def sphere_density( XPos,YPos,ZPos,Density,Radius,XScale,YScale):    
    Mass = (4/3)*np.pi*np.power(Radius,3)*Density                
    matrix = (Kappa*Mass*ZPos)/np.power((np.power((XScale-XPos),2) + np.power((YScale-YPos),2) + ZPos*ZPos),(1.5))  
    return matrix

def sphere_mass(XPos,YPos,ZPos,Mass,Radius,XScale,YScale):
    matrix = (Kappa*Mass*ZPos)/np.power((np.power((XScale-XPos),2) + np.power((YScale-YPos),2) + ZPos*ZPos),(1.5))  
    return matrix

def cylinder_vertical(XPos,YPos,ZPos,Density,Radius,XScale,YScale):
    Mass = 2*np.pi*np.power(Radius,2)*Density        
    matrix = (Kappa*Mass)/np.power((np.power((XScale-XPos),2) + np.power((YScale-YPos),2) + ZPos*ZPos),(0.5))
    return matrix

def random_data(DataScale,XLength,YLength):
    matrix = DataScale*np.random.rand(XLength,YLength)
    return matrix

def cylinder_horizontal(XPos,YPos,XPos2,YPos2,ZPos,Density,Radius,XScale,YScale):
    Mass = 2*np.pi*np.power(Radius,2)*Density    
    u1 = XPos2 - XPos
    u2 = YPos2 - YPos
    n1 = -1*u2
    n2 = u1
    c = -1*(n1*XPos + n2*YPos)
    matrix = 2*Kappa*Mass*ZPos/(np.power(np.absolute(n1*XScale + n2*YScale + c)/(np.sqrt(n1*n1 + n2*n2)),2) + ZPos*ZPos)                        
      
    return matrix

    
    
def cube(XPos,YPos,XPos2,YPos2,ZPos,ZPos2,Density,Radius,XPoints,YPoints):
# TO BE REVISED, THE DATA ARE NOT CORRESPONDING TO ANOMALY DEFINITION!    
#==============================================================================
# % RECTANGULAR PRISM according to Blakely, pg 187
# % GIVEN BY x1,y1,y1 (top left corner) and x2,y2,z2 (right deep corner)
# % plus density
# 
#                     
#     CKappa=6.670E-11;   
#    
# %second part count the matrice of influence of the anomaly object       
#     for i = 1:length(X) - 1
#         for j = 1:length(Y) - 1            
# % BLAKELY  -----------          
#             value = 0;
# %            move the observation point to the center of coordinates
#            rx = [(x1 - i) (x2 - i)];
#            ry = [(y1 - j) (y2 - j)];
#            rz = [z1 z2];
#             
#             
# 
#             for x = 1:2
#                 for y = 1:2
#                     for z = 1:2
#                         my = power(-1,x)*power(-1,y)*power(-1,z);
#                         R = sqrt(rx(x)^2 + ry(y)^2 + rz(z)^2);
#                         value = value + my*(rz(z)*atan((rx(x)*ry(y))/(rz(z)*R)) - rx(x)*log(R+ry(y)) - ry(y)*log(R+rx(x)));
#                     end
#                 end
#             end
# % END OF BLAKELY IMPLEMENTATION
#==============================================================================
    MaxX = np.size(XPoints)
    MaxY = np.size(YPoints)
    dim = (MaxX,MaxY)
    matrix = np.zeros(dim)


    rz = np.array([ZPos,ZPos2])
    for i in range(MaxX):
        for j in range(MaxY):
            value = 0
            rx = np.array([XPos - XPoints[i],XPos2 - XPoints[i]])
            ry = np.array([YPos - YPoints[j],YPos2 - YPoints[j]])
            
            for x in range(1,3):
                for y in range(1,3):
                    for z in range(1,3):
                        my = np.power(-1,x)*np.power(-1,y)*np.power(-1,z)
                        R = np.sqrt(rx[x-1]*rx[x-1]+ry[y-1]*ry[y-1]+rz[z-1]*rz[z-1])
                        value = value + my*(rz[z-1]*np.arctan((rx[x-1]*ry[y-1])/(rz[z-1]*R)) - rx[x-1]*np.log(R+ry[y-1]) - ry[y-1]*np.log(R+rx[x-1]))
            value = value*Kappa*Density
            matrix[i,j] = value
    return matrix

#==============================================================================
# Tohle neni jeste dodelano, bude se na tom pracovat pres leto
#==============================================================================

def cylinder_sloped(XPos,YPos,XPos2,YPos2,ZPos,ZPos2,Density,Radius,XScale,YScale):
    matrix=[]
    return matrix

