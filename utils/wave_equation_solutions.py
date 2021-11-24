import numpy as np
from models.datastructures import BoundaryType

def WaveEquation1D(grid, x0, boundary_cond, c, sigma0, num_reflections=4):
    """ Analytical solution with Dirichlet or Neumann boundaries
        num_reflections: number of reflections in the solution
    """
    assert(boundary_cond == BoundaryType.DIRICHLET or boundary_cond == BoundaryType.NEUMANN)

    amp_sign = 1

    x_mesh = grid[0]
    t_mesh = grid[1]

    xmin = np.min(x_mesh)
    xmax = np.max(x_mesh)
    
    L = ( xmax - xmin )

    # 1D wave function definition: x0 is an array containing 2 elements for positive and negative travelling waves, respectively
    pfunc = lambda x,t,x0: 0.5*np.exp(-0.5*(np.array((x-x0[0] - c*t))/sigma0)**2) + \
                           0.5*np.exp(-0.5*(np.array((x-x0[1] + c*t))/sigma0)**2)

    # initial wave solution (no reflections)
    p = pfunc(x_mesh,t_mesh,[x0,x0])

    if num_reflections <= 0:
        return p

    # calculate starting positions for reflected waves for superposition
    x0_rel = (x0 - xmin) / L # relative position

    for i in range(num_reflections):
        if np.mod(i,2) == 0:
            x0_min = xmin - i*L - L*x0_rel       # x0 for positive travelling wave
            x0_max = xmax + (i+1)*L - L*x0_rel   # x0 for negative travelling wave
        else:
            x0_min = xmin - i*L - (L - L*x0_rel)  # x0 for positive travelling wave
            x0_max = xmax + i*L + L*x0_rel        # x0 for negative travelling wave

        if boundary_cond == BoundaryType.DIRICHLET:
            amp_sign = -1*amp_sign
                                 
        p = p + amp_sign*pfunc(x_mesh,t_mesh,[x0_min,x0_max])
            
    return p

def WaveEquation2D(grid, X0, boundary_cond, c, sigma0, num_reflections=4):
    """ Analytical solution with Dirichlet or Neumann boundaries
        num_reflections: number of reflections in the solution
    """
    assert(boundary_cond == BoundaryType.DIRICHLET or boundary_cond == BoundaryType.NEUMANN)

    amp_sign = 1

    x_mesh = grid[0]
    y_mesh = grid[1]
    t_mesh = grid[2]

    xmin = np.min(x_mesh)
    xmax = np.max(x_mesh)
    ymin = np.min(y_mesh)
    ymax = np.max(y_mesh)
    
    x0 = X0[0]
    y0 = X0[1]

    Lx = ( xmax - xmin )
    Ly = ( ymax - ymin )

    # 2D wave function definition: x0 and y0 are arrays containing 2 elements each for positive and negative travelling waves, respectively
    pfunc = lambda x,y,t,x0,y0: 0.5*np.exp(-0.5*(np.array((x-x0[0] - c*t))/sigma0)**2)*np.exp(-0.5*(np.array((y-y0[0] - c*t))/sigma0)**2) + \
                                0.5*np.exp(-0.5*(np.array((x-x0[1] + c*t))/sigma0)**2)*np.exp(-0.5*(np.array((y-y0[1] + c*t))/sigma0)**2)

    # initial wave solution (no reflections)
    p = pfunc(x_mesh,y_mesh,t_mesh,[x0,x0],[y0,y0])

    if num_reflections <= 0:
        return p

    # calculate starting positions for reflected waves for superposition
    x0_rel = (x0 - xmin) / Lx # relative position
    y0_rel = (y0 - ymin) / Ly # relative position

    for i in range(num_reflections):
        if np.mod(i,2) == 0:
            # x0 for positive travelling wave
            x0_min = xmin - i*Lx - Lx*x0_rel      
            y0_min = ymin - i*Ly - Ly*y0_rel

            # x0 for negative travelling wave
            x0_max = xmax + (i+1)*Lx - Lx*x0_rel
            y0_max = ymax + (i+1)*Ly - Ly*y0_rel
        else:
            # x0 for positive travelling wave
            x0_min = xmin - i*Lx - (Lx - Lx*x0_rel)
            y0_min = ymin - i*Ly - (Ly - Ly*y0_rel)

            # x0 for negative travelling wave
            x0_max = xmax + i*Lx + Lx*x0_rel
            y0_max = ymax + i*Ly + Ly*y0_rel

        if boundary_cond == BoundaryType.DIRICHLET:
            amp_sign = -1*amp_sign

        p = p + amp_sign*pfunc(x_mesh,y_mesh,t_mesh,[x0_min,x0_max],[y0_min,y0_max])

    return p

def generateSolutionData(grids, x0_sources, c, sigma0, boundary_cond):
    p_data = []    

    spatial_dim = np.asarray(x0_sources).shape[1]
    for i, x0 in enumerate(x0_sources):
        if spatial_dim == 1:
            p_sol = WaveEquation1D(grids[i],x0,boundary_cond,c,sigma0)
        elif spatial_dim == 2:
            p_sol = WaveEquation2D(grids[i],x0,boundary_cond,c,sigma0)
        else:
            raise NotImplementedError()

        p_data.append(np.asarray(p_sol)) # NBJ reshape removed

    return np.asarray(p_data)